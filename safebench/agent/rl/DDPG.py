import os
import sys
import copy
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer

sys.path.append('..')
from utils import CUDA, hidden_init


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(Actor, self).__init__()
        self.scale = scale
        l1_size = 128
        l2_size = 128
        self.l1 = nn.Linear(state_dim, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = x/self.scale
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(Critic, self).__init__()
        self.scale = scale
        l1_size = 128
        l2_size = 128
        self.l1 = nn.Linear(state_dim, l1_size)
        self.l2 = nn.Linear(l1_size+action_dim, l2_size)
        self.l3 = nn.Linear(l2_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x, u):
        x = x/self.scale
        xs = F.relu(self.l1(x))
        x = torch.cat([xs, u], dim=1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    ''' DDPG is an on-policy algorithm, so we store the risk data and safe data separately. '''
    name = 'DDPG'
    
    def __init__(self, args):
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.actor_lr = args['actor_lr']
        self.critic_lr = args['critic_lr']
        self.tau = args['tau']
        self.gamma = args['gamma']
        self.memory_capacity = args['memory_capacity']
        self.batch_size = args['batch_size']
        self.update_iteration = args['update_iteration']
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        self.scale = args['scale']
        self.epsilon = args['epsilon']
        self.risk_aware = args['risk_aware']
        
        self.actor = CUDA(Actor(self.state_dim, self.action_dim, self.scale))
        self.actor_target = CUDA(Actor(self.state_dim, self.action_dim, self.scale))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = CUDA(Critic(self.state_dim, self.action_dim, self.scale))
        self.critic_target = CUDA(Critic(self.state_dim, self.action_dim, self.scale))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # buffer for saving data
        self.replay_buffer = ReplayBuffer(self.memory_capacity, self.state_dim*2+self.action_dim+2, self.risk_aware)

    def store_transition(self, data):
        risk = data[-1]
        # preprocess the data before store t into buffer
        # [state_a, action, post_reward, next_state_a, done, risk]
        data = np.concatenate([data[0], data[1], [data[2]], data[3], [np.float(data[4])]])
        self.replay_buffer.push(data, risk)

    def select_action(self, state, deterministic=False):
        if np.random.randn() > self.epsilon or deterministic: # greedy policy
            state = CUDA(torch.FloatTensor(state.reshape(1, -1)))
            action = self.actor(state).cpu().data.numpy().flatten()
        else: # random policy
            action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        
        # decay epsilon
        self.epsilon *= 0.99

        return action

    def train(self):
        # check if memory is enough for one batch
        if self.replay_buffer.memory_len < self.batch_size:
            return

        for it in range(self.update_iteration):
            # Sample replay buffer
            batch_memory = self.replay_buffer.sample(self.batch_size)
            state = CUDA(torch.FloatTensor(batch_memory[:, 0:self.state_dim]))
            action = CUDA(torch.FloatTensor(batch_memory[:, self.state_dim:self.state_dim+self.action_dim]))
            reward = CUDA(torch.FloatTensor(batch_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1]))
            next_state = CUDA(torch.FloatTensor(batch_memory[:, self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+1]))
            done = CUDA(torch.FloatTensor(1-batch_memory[:, 2*self.state_dim+self.action_dim+1:2*self.state_dim+self.action_dim+2]))

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self):
        states = {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
        filepath = os.path.join(self.model_path, 'model.ddpg.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.ddpg.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
        else:
            raise Exception('No DDPG model found!')

