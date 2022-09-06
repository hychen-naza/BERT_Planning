"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
from mingpt.bert_model_babyai import token2idx
from copy import deepcopy

AGENT_ID = 10
AGENT_COLOR = 6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def bert_sample_multi_step(bert_model, x, y=None, timesteps=None, rtgs=None, insts=None, logger=None, sample_iteration=1):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    bert_model.eval()
    batch_size = x.shape[0]
    context = bert_model.config.context
    plan_horizon = bert_model.config.plan_horizon
    PAD_ACTION = bert_model.config.vocab_size-1

    sampled_actions = []

    context_obss = torch.clone(x[:, -context:])
    init_obss = torch.clone(x[:,-1]).cpu()
    sample_obss = torch.repeat_interleave(torch.Tensor(init_obss).unsqueeze(1), plan_horizon-1, dim=1).to(dtype=torch.float32).to('cuda')
    sample_obss = torch.cat((context_obss, sample_obss), dim=1).to(dtype=torch.float32)

    context_rtgs = torch.clone(rtgs[:, -context:])
    sample_rtgs = torch.tensor([rtgs[0,-1,0].cpu().item()]*(plan_horizon-1), dtype=torch.float32).to('cuda').unsqueeze(0).unsqueeze(-1)
    sample_rtgs = torch.cat((context_rtgs, sample_rtgs), dim=1).to(dtype=torch.float32)

    sample_actions = [[PAD_ACTION] for i in range(plan_horizon)]
    sample_actions = torch.repeat_interleave(torch.Tensor(sample_actions).unsqueeze(0), batch_size, dim=0).to(dtype=torch.long).to('cuda')
    if (y is not None):
        context_actions = torch.clone(y[:, -context+1:])

    init_temperature = 0.5
    energys = []
    # MCMC construct sample trajectory
    for i in range(sample_iteration):
        temperature = min(init_temperature + i / (2*sample_iteration), 0.9) if i > 0 else 0
        action_masks = np.random.uniform(0, 1, (batch_size, plan_horizon, 1)) >= temperature
        action_masks = torch.from_numpy(action_masks).to('cuda')
        sample_actions[action_masks] = PAD_ACTION #token2idx('<-MASK->')
        # sample_actions = torch.cat((context_actions, sample_actions), dim=1).to(dtype=torch.long)
        
        input_actions = torch.cat((context_actions, sample_actions), dim=1).to(dtype=torch.long) if y is not None else sample_actions
        action_logits = bert_model.forward(sample_obss, input_actions, rtgs=sample_rtgs, timesteps=timesteps)

        _, action_val = torch.topk(F.softmax(action_logits.reshape(-1,action_logits.shape[2]), dim=-1), k=1, dim=-1)
        action_val = action_val.reshape(batch_size, input_actions.shape[1])[:,-plan_horizon:]
        if (i < sample_iteration-1):
            sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
            '''
            iter_actions = sample_actions.cpu().numpy()
            update_states, update_obss = bert_model.update_samples(sample_states[:,-plan_horizon:], sample_obss[:,-plan_horizon:], iter_actions)
            sample_states = torch.cat((context_states[:,:-1], update_states), dim=1).to(dtype=torch.long)
            sample_obss = torch.cat((context_obss[:,:-1], update_obss), dim=1).to(dtype=torch.float32)
            '''
            #energys.append(self.compute_energy(target_actions, sample_actions, action_logits, torch.logical_and(action_masks, target_action_masks)).item())
        else:
            train_sample_actions = sample_actions.clone()
            train_sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
            #sample_actions[action_masks] = action_val.unsqueeze(2)[action_masks]
            #iter_actions = train_sample_actions.cpu().numpy()
            #sample_states = self.update_sample_states(sample_states, iter_actions)
            #energys.append(self.compute_energy(target_actions, train_sample_actions, action_logits, torch.logical_and(action_masks, target_action_masks)).item())
    bert_model.train()
    return train_sample_actions[0,-plan_horizon:,0].flatten().cpu().tolist()