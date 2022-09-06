"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import gym
import logging
from tqdm import tqdm
import numpy as np
import pdb
import torch
import sys
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import babyai.utils as utils

logger = logging.getLogger(__name__)

from mingpt.utils import bert_sample_multi_step, AGENT_ID, AGENT_COLOR
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import logging
from babyai.utils.agent import BotAgent
from copy import deepcopy

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.ckpt_path = f'./model_checkpoint/{self.game}_{self.seed}_best_checkpoint.pth'

class Trainer:

    def __init__(self, bert_model, bert_train_dataset, test_dataset, config, rate, plan_horizon, sample_iteration):
        self.bert_model = bert_model
        self.bert_train_dataset = bert_train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.sample_iteration = sample_iteration


        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.bert_model = self.bert_model.to(self.device)
            #self.dt_model = torch.nn.DataParallel(self.dt_model).to(self.device)
            #self.bert_model = torch.nn.DataParallel(self.bert_model).to(self.device)
        console = logging.StreamHandler(sys.stdout)
        console_log_level = 100
        console.setLevel(console_log_level)
        self.logger = logging.getLogger(__name__)  
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f'log_file_{self.plan_horizon}.log')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def save_checkpoint(self):
        print(f"saving {self.config.ckpt_path}")
        torch.save(self.bert_model, self.config.ckpt_path)

    def load_checkpoint(self):
        print(f"loading {self.config.ckpt_path}")
        self.bert_model = torch.load(self.config.ckpt_path)
        self.bert_model.eval()

    def train(self):
        bert_model, config = self.bert_model, self.config

        raw_bert_model = bert_model.module if hasattr(self.bert_model, "module") else bert_model
        bert_optimizer = raw_bert_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            bert_model.train(is_train)
            bert_data = self.bert_train_dataset if is_train else self.test_dataset
            bert_loader = DataLoader(bert_data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=0) #config.num_workers
            bert_losses = []
            gt_traj_energys = []
            mcmc_energys = []
            free_rates = []
            action_correct_rates = []
            all_action_correct_rate_steps = []
            pos_energies, neg_energies = [], []

            pbar = tqdm(enumerate(bert_loader), total=len(bert_loader)) if is_train else enumerate(bert_loader)
            for it, (x, y, r, msk_y, t, init_x) in pbar:#enumerate(bert_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                msk_y = msk_y.to(self.device)
                t = t.to(self.device)

                with torch.set_grad_enabled(is_train):
                    bert_loss, gt_traj_energy, mcmc_energy, \
                        action_correct_rate, action_correct_rate_steps, pos_energy, neg_energy \
                        = bert_model.train_step(x, y, action_masks=msk_y,\
                            timesteps=t,init_states=init_x, rtgs=r, logger = self.logger) 
                    bert_loss = bert_loss.mean() 
                    bert_losses.append(bert_loss.item())
                    gt_traj_energys.append(gt_traj_energy)
                    mcmc_energys.append(mcmc_energy)
                    action_correct_rates.append(action_correct_rate)
                    all_action_correct_rate_steps.append(action_correct_rate_steps)
                    pos_energies.append(pos_energy.item())
                    neg_energies.append(neg_energy.item())

                    bert_model.zero_grad()
                    bert_loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), config.grad_norm_clip, error_if_nonfinite = True)
                    except:
                        # Non-finite norm encountered in torch.nn.utils.clip_grad_norm_;
                        # don't do gradient descend at this case
                        continue
                    bert_optimizer.step()

                
                if is_train:
                    # backprop and update the parameters in model
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in bert_optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: bert loss {bert_loss.item():.5f}. lr {lr:e}")
                interval = 300
                # very initial result
                #if (epoch_num ==0 and it == 2):
                #    msg = f'Test it {it}, epoch_num {epoch_num}, bert loss {np.mean(bert_losses):.5f}, gt_traj_energys {np.mean(gt_traj_energys):.5f}, mcmc_better_than_first_rates {np.mean(mcmc_better_than_first_rates):.5f}, mcmc__better_than_all_rates {np.mean(mcmc__better_than_all_rates):.5f}, mcmc_energys {np.mean(mcmc_energys):.5f}, action_correct_rate {np.mean(action_correct_rates):.5f}, all_action_correct_rate_steps {np.mean(all_action_correct_rate_steps[-interval:],axis=0)}, pos eng {np.mean(pos_energies[-interval:]):.5f}, neg eng {np.mean(neg_energies[-interval:]):.5f}' #rates {np.mean(rates)}, 
                #    self.logger.info(msg)
                # it % interval == 0 and 
                #if (it == 0 and epoch_num >= 1 and epoch_num % 1 ==0): #    % 4000  and it > 100  it >= 2
                    #msg = f'Test it {it}, epoch_num {epoch_num}, bert loss {np.mean(bert_losses[-interval:]):.5f}, gt_traj_energys {np.mean(gt_traj_energys[-interval:]):.5f}, mcmc_energys {np.mean(mcmc_energys[-interval:]):.5f}, action_correct_rate {np.mean(action_correct_rates[-interval:]):.5f}, all_action_correct_rate_steps {np.mean(all_action_correct_rate_steps[-interval:],axis=0)}, pos eng {np.mean(pos_energies[-interval:]):.5f}, neg eng {np.mean(neg_energies[-interval:]):.5f}' #rates {np.mean(rates)}, 
                    #self.logger.info(msg)
                    #self.test_returns(0, test_num=5) 
                    #if (np.mean(action_correct_rates[-interval:])>0.95):
                    #    pdb.set_trace()
                    #tmp = np.mean(all_action_correct_rate_steps[-interval:],axis=0)
                    #if (np.isnan(tmp[-1])):
                    #    pdb.set_trace()
        self.tokens = 0 # counter used for learning rate decay
        best_eval_return = 0
        if self.config.game == 'Breakout':
            ret = 90
        elif self.config.game == 'Seaquest':
            ret = 1150
        elif self.config.game == 'Qbert':
            ret = 14000
        elif self.config.game == 'Pong':
            ret = 20
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)  
            eval_return = self.test_returns(ret, test_num=10) 
            if (eval_return > best_eval_return):
                best_eval_return = eval_return
                self.save_checkpoint()
        self.load_checkpoint()
        self.test_returns(ret, test_num=40) 

    def test_returns(self, ret, test_num=10):
        self.bert_model.train(False)
        self.bert_model.eval()
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()
        T_rewards = []
        done = True
        success_count = 0
        for i in range(test_num):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sample_actions = bert_sample_multi_step(self.bert_model, state, \
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device), \
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), \
                logger=self.logger, sample_iteration=self.sample_iteration)

            all_states = state
            actions = []
            j = 0
            reward_sum = 0
            done = False

            while True:
                for action in sample_actions[:1]: #[:1]
                    #action = 3 #random.randint(0, 3)
                    state, reward, done = env.step(action)
                    #print(f"action {action}, reward {reward}, done {done}, len {len(actions)}, zero count {actions.count(0)}")
                    #pdb.set_trace()
                    actions.append(action)
                    reward_sum += reward
                    rtgs += [rtgs[-1] - reward]
                    j += 1
                    state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                    all_states = torch.cat([all_states, state], dim=1)
                    if (done):
                        #print(f"actions {actions}, length {len(actions)}")
                        break
                if (done):
                    T_rewards.append(reward_sum)
                    zero_lengths = actions.count(0)
                    #print(f"length {len(actions)}, reward_sum {reward_sum}, occurrence [{actions.count(0)},{actions.count(1)}, {actions.count(2)}, {actions.count(3)}]")
                    break
                sample_actions = bert_sample_multi_step(self.bert_model, all_states, y=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),\
                    logger=self.logger,\
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), \
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)), \
                    sample_iteration=self.sample_iteration)

        env.close()
        eval_return = sum(T_rewards)/float(test_num)
        success_rate = success_count /float(test_num)
        msg = f"eval ret {ret} return test_num {test_num}: {eval_return}, success_rate: {success_rate:.3f}"
        self.logger.info(msg)
        self.bert_model.train(True)
        return eval_return

class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 10e3
        self.game = game
        self.history_length = 4
