'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:57:03
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import gym
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import sys
import os

sys.path.append('..')
from utils import CUDA, CPU, kaiming_init


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, scale, std=0.0):
        super(ActorCritic, self).__init__()
        self.scale = scale

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_outputs),
            nn.Softmax(dim=1),
        )

        # init parameters
        self.apply(kaiming_init)

    def forward(self, x):
        x = x/self.scale
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value


class A2C():
    ''' A2C is an on-policy algorithm, which cannot store risky buffer. '''
    name = 'A2C'

    def __init__(self, args):
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.action_dim = args['action_dim']
        self.hidden_dim = args['hidden_dim']
        self.gamma = args['gamma']
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        self.scale = args['scale']

        self.model = CUDA(ActorCritic(self.state_dim, self.action_dim*2, self.hidden_dim, self.scale))
        self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'])

        self.entropy = 0
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.returns = []
        self.action = None
        self.dist = None
        self.value = None

    def reset_buffer(self):
        self.entropy = 0
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.returns = []
        self.action = None
        self.dist = None
        self.value = None

    def select_action(self, state, determinstic=False):
        state = CUDA(torch.FloatTensor(state[None]))
        self.dist, self.value = self.model(state)
        if determinstic:
            self.action = torch.argmax(self.dist.probs[0])[None]
        else:
            self.action = self.dist.sample()
        return CPU(self.action)[0]

    # call after select_action
    def store_transition(self, reward, state_a, action):
        # state_a, action are useless, just to be in line with PPO
        # store the results
        self.entropy += self.dist.entropy().mean()
        self.log_probs.append(self.dist.log_prob(self.action))
        self.values.append(self.value[0])
        self.rewards.append(CUDA(torch.FloatTensor([reward])))

    def compute_returns(self, last_state):
        last_state = CUDA(torch.FloatTensor(last_state[None]))
        _, last_value = self.model(last_state)
        R = last_value[0]
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R
            self.returns.insert(0, R)

    def train(self, last_state):
        # compute the reward of current sequence
        self.compute_returns(last_state)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(self.returns).detach()
        values    = torch.cat(self.values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # since A2C is an on-policy method, old data cannot be used
        self.reset_buffer()
        return CPU(loss)

    def save_model(self):
        states = {'parameters': self.model.state_dict()}
        filepath = os.path.join(self.model_path, 'model.a2c.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.a2c.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        else:
            raise Exception('No A2C model found!')
