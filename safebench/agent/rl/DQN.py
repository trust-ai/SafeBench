'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:55:23
Description: 
    Copyright (c) 2022-2023 Safebench Team

    Modified from <https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch>

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from buffer import ReplayBuffer

sys.path.append('..')
from utils import CUDA, CPU, kaiming_init


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, state_dim, action_dim, hidden_dim, scale):
        super(Net, self).__init__()
        self.scale = scale

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(kaiming_init)

    def forward(self, x):
        x = x/self.scale
        action_prob = self.model(x)
        return action_prob


class DQN():
    ''' DQN is an off-policy algorithm, so we store the risk data and safe data separately. '''
    name = 'DQN'

    def __init__(self, args):
        super(DQN, self).__init__()

        self.lr = args['lr']
        self.gamma = args['gamma']
        self.epsilon = args['epsilon']
        self.memory_capacity = args['memory_capacity']
        self.q_network_iteration = args['q_network_iteration']
        self.batch_size = args['batch_size']
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        self.hidden_dim = args['hidden_dim']
        self.scale = args['scale']
        self.risk_aware = args['risk_aware']

        # create two models
        self.eval_net = CUDA(Net(self.state_dim, self.action_dim*2, self.hidden_dim, self.scale))
        self.target_net = CUDA(Net(self.state_dim, self.action_dim*2, self.hidden_dim, self.scale))

        # [state, action, reward, next_state]
        # NOTE: here the action dimension is 1, because we only store the index not the probability
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.state_dim*2+self.action_dim+1, self.risk_aware)
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0

    def select_action(self, state, determinstic=False):
        state = CUDA(torch.FloatTensor(state))
        if np.random.randn() > self.epsilon or determinstic: # greedy policy
            action_value = self.eval_net.forward(state)
            action = CPU(torch.max(action_value, 0)[1])
        else: # random policy
            action = np.random.randint(0, self.action_dim*2)
        
        # decay epsilon
        self.epsilon *= 0.99

        return action

    def store_transition(self, data):
        state, action, reward, next_state, _, risk = data
        data = np.concatenate([state, [action], [reward], next_state])
        self.replay_buffer.push(data, risk)

    def train(self):
        # check if memory is full
        if self.replay_buffer.memory_len < self.memory_capacity:
            return False

        #update the parameters
        if self.learn_step_counter % self.q_network_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #sample batch from memory
        batch_memory = self.replay_buffer.sample(self.batch_size)
        batch_state = CUDA(torch.FloatTensor(batch_memory[:, :self.state_dim]))
        batch_action = CUDA(torch.LongTensor(batch_memory[:, self.state_dim:self.state_dim+self.action_dim].astype(int)))
        batch_reward = CUDA(torch.FloatTensor(batch_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1]))
        batch_next_state = CUDA(torch.FloatTensor(batch_memory[:, -self.state_dim:]))

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # no gradient required
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.save_model()
        return True

    def save_model(self):
        states = {'parameters': self.eval_net.state_dict()}
        filepath = os.path.join(self.model_path, 'model.dqn.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.dqn.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.eval_net.load_state_dict(checkpoint['parameters'])
        else:
            raise Exception('No DQN model found!')
