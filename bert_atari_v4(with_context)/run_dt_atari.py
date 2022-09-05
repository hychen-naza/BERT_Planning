import csv
import logging
from xmlrpc.client import boolean
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.dt_model_babyai import DT_GPT, GPTConfig
from mingpt.bert_model_babyai import BERT_GPT, token2idx, tokens
from mingpt.trainer_atari import Trainer, TrainerConfig
from collections import deque
import random
import torch
import pickle
import gym
import blosc
import pdb
import copy
import argparse
from create_dataset import create_dataset
import babyai.utils as utils
from instruction_process import InstructionsPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--sample_iteration', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--data_dir_prefix', type=str, default='/local/datasets/dqn/')
args = parser.parse_args()

set_seed(args.seed)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class BERTDataset(Dataset):
    def __init__(self, data, actions, done_idxs, rtgs, timesteps, rate, plan_horizon, context):        
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

        self.vocab_size = max(actions) + 1 + 1
        self.rate = rate
        self.plan_horizon = plan_horizon
        self.context = context

    def __len__(self):
        return len(self.data) - (self.context - self.plan_horizon)*self.rate

    def get_init_states(self, states):
        return torch.clone(states[-1])

    def __getitem__(self, idx):
        block_size = self.plan_horizon + self.context
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        context_start_idx = done_idx - block_size
        context_end_idx = done_idx - block_size + self.context

        #states = torch.tensor(np.array(self.data[context_start_idx:context_end_idx]), dtype=torch.float32).reshape(self.context, -1) # (block_size, 4*84*84)
        states = torch.tensor(np.array(self.data[context_start_idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        init_state = torch.tensor(np.array(self.data[context_end_idx]), dtype=torch.float32).reshape(-1)

        actions = torch.tensor(self.actions[context_start_idx:done_idx], dtype=torch.long).unsqueeze(1) 
        action_msk = np.ones((block_size,1)) #np.concatenate([np.zeros((self.context,1)), np.ones((self.plan_horizon,1))], axis=0)
        action_msk = torch.from_numpy(action_msk).to(dtype=torch.bool)

        timesteps = torch.tensor(self.timesteps[context_start_idx:context_start_idx+1], dtype=torch.int64).unsqueeze(1)
        return states, actions, action_msk, timesteps, init_state

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

rate = 3 if args.model_type == 'reward_conditioned' else 2
plan_horizon = args.horizon
context = args.context_length

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)
bert_train_dataset = BERTDataset(obss, actions, done_idxs, returns, timesteps, rate, plan_horizon, context)

mconf = GPTConfig(bert_train_dataset.vocab_size, (context+plan_horizon)*rate,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps), \
                    sample_iteration = args.sample_iteration, context=context, plan_horizon = plan_horizon)
bert_model = BERT_GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(bert_train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
trainer = Trainer(bert_model, bert_train_dataset, None, tconf, rate, plan_horizon, args.sample_iteration)

trainer.train()
