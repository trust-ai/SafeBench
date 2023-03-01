'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:55:23
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/gouxiangchen/ac-ppo>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from safebench.util.torch_util import CUDA, CPU, hidden_init


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_std = nn.Linear(128, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.min_val = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
        self.fc_std.weight.data.uniform_(*hidden_init(self.fc_std))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + self.min_val
        return mu, std

    def select_action(self, state, deterministic):
        with torch.no_grad():
            mu, std = self.forward(state)
            if deterministic:
                action = mu
            else:
                n = Normal(mu, std)
                action = n.sample()
        return CPU(action[0])


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO():
    name = 'PPO'
    type = 'onpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.gamma = config['gamma']
        self.policy_lr = config['policy_lr']
        self.value_lr = config['value_lr']
        self.train_iteration = config['train_iteration']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.model_path = config['model_path']
        self.model_id = config['model_id']
        self.clip_epsilon = config['clip_epsilon']

        self.policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.old_policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = CUDA(ValueNetwork(state_dim=self.state_dim))
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)
        self.reset_buffer()

    def reset_buffer(self):
        # reset buffer
        self.rewards = []
        self.states = []
        self.actions = []

    def select_action(self, state, deterministic=False):
        state_tensor = CUDA(torch.FloatTensor(state).unsqueeze(0))
        action = self.policy.select_action(state_tensor, deterministic)
        return action

    def train(self, next_state):
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # process the last state
        with torch.no_grad():
            next_state_tensor = CUDA(torch.FloatTensor(next_state).unsqueeze(0))
            R = self.value(next_state_tensor)
        # computer rewards
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            self.rewards[i] = R
        rewards_tensor = CUDA(torch.FloatTensor(self.rewards).unsqueeze(1))
        
        # start to train, use gradient descent without batch size
        for K in range(self.train_iteration):
            state_tensor = CUDA(torch.FloatTensor(self.states))
            action_tensor = CUDA(torch.FloatTensor(self.actions))
            with torch.no_grad():
                advantage = rewards_tensor - self.value(state_tensor)
                old_mu, old_std = self.old_policy(state_tensor)
                old_n = Normal(old_mu, old_std)

            mu, std = self.policy(state_tensor)
            n = Normal(mu, std)
            log_prob = n.log_prob(action_tensor)
            old_log_prob = old_n.log_prob(action_tensor)
            ratio = torch.exp(log_prob - old_log_prob)
            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
            loss = torch.min(L1, L2)
            loss = -loss.mean()
            
            # update parameters
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            value_loss = F.mse_loss(rewards_tensor, self.value(state_tensor))
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        
        # reset buffer
        self.reset_buffer()

    def save_model(self):
        states = {'policy': self.policy.state_dict()}
        filepath = os.path.join(self.model_path, 'model.ppo_gae.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.ppo_gae.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.policy.load_state_dict(checkpoint['policy'])
        else:
            raise Exception('No PPO model found!')
