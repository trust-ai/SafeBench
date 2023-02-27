'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
@LastEditTime: 2020-07-13 14:50:33
@Description: 
    A PPO implementation with Generalized Advantage Estimator (GAE) to improve the sample efficiency
    Modified from https://github.com/gouxiangchen/ac-ppo
'''

import gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import copy
import sys
import os

sys.path.append('../')
from utils import CUDA, CPU, hidden_init


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_std = nn.Linear(128, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.scale = scale
        self.min_val = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_mu.weight.data.uniform_(*hidden_init(self.fc_mu))
        self.fc_std.weight.data.uniform_(*hidden_init(self.fc_std))

    def forward(self, x):
        x = x/self.scale
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
    def __init__(self, state_dim, scale):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = x/self.scale
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO_GAE():
    ''' PPO is an on-policy method, which doesn't need to store buffer '''
    name = 'PPO_GAE'

    def __init__(self, args):
        self.gamma = args['gamma']
        self.policy_lr = args['policy_lr']
        self.value_lr = args['value_lr']
        self.train_iteration = args['train_iteration']
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        self.scale = args['scale']
        self.clip_epsilon = args['clip_epsilon']

        self.policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim, scale=self.scale))
        self.old_policy = CUDA(PolicyNetwork(state_dim=self.state_dim, action_dim=self.action_dim, scale=self.scale))
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value = CUDA(ValueNetwork(state_dim=self.state_dim, scale=self.scale))
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

    # call after choose_action
    def store_transition(self, reward, state, action):
        # store the results
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)

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
            raise Exception('No PPO_GAE model found!')


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    episodes = 1000
    args = {
        'state_dim': 3,
        'action_dim': 1,
        'gamma': 0.9,
        'train_iteration': 10,
        'policy_lr': 1e-5,
        'value_lr': 2e-5,
        'train_interval': 32,
    }
    agent = PPO_GAE(args)

    for e_i in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        s_i = 0
        while not done:
            action = 2.0*agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            reward = (reward + 8.1) / 8.1

            agent.store_transition(reward, state, action)
            state = copy.deepcopy(next_state)

            if (s_i+1) % args['train_interval'] == 0 or done:
                agent.train(state)
            s_i += 1

        if e_i % 10 == 0:
            print('[{}/{}] episode_reward: {}'.format(e_i, episodes, episode_reward))
