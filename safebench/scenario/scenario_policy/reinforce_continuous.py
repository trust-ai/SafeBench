'''
@Author: 
@Email: 
@Date: 2020-01-24 13:52:10
LastEditTime: 2023-02-27 21:49:13
@Description: 
'''

import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.util.torch_util import CUDA, CPU


def constraint(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x


def normalize_routes(routes):
    mean_x = np.mean(routes[:, 0:1])
    max_x = np.max(np.abs(routes[:, 0:1]))
    x_1_2 = (routes[:, 0:1] - mean_x) / max_x

    mean_y = np.mean(routes[:, 1:2])
    max_y = np.max(np.abs(routes[:, 1:2]))
    y_1_2 = (routes[:, 1:2] - mean_y) / max_y

    route = np.concatenate([x_1_2, y_1_2], axis=0)
    return route


pi = CUDA(torch.FloatTensor([math.pi]))
def normal(x, mu, sigma_sq):
    a = (-1*(CUDA(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class AutoregressiveModel(nn.Module):
    def __init__(self, standard_action=True, num_waypoint=20):
        super(AutoregressiveModel, self).__init__()
        self.standard_action = standard_action
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 32

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1

        # TODO: remove this parameter
        if self.standard_action:
            self.d_os = 1

        self.fc_input = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(inplace=True)
        )

        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))

        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))

        self.fc_action_c = nn.Sequential(
            nn.Linear(1+1+hidden_size_1, self.c_os*2),
        )

        if self.standard_action:
            self.fc_action_d = nn.Sequential(
                nn.Linear(1+1+1+hidden_size_1, self.d_os*2),
            )

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        #mu = torch.tanh(normal_action[:, :action_os])
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = CUDA(Variable(torch.randn(mu.size())))
        action = (mu + sigma*eps)
        # action = (mu + sigma.sqrt()*eps)

        return action, mu, sigma

    def forward(self, x):
        # p(s)
        s = self.fc_input(x)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s)
        state_sample_a = torch.cat((s, action_a), dim=1)
        normal_b = self.fc_action_b(state_sample_a)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, action_a, action_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        if self.standard_action:
            state_sample_a_b_c = torch.cat((s, action_a, action_b, action_c), dim=1)
            normal_d = self.fc_action_d(state_sample_a_b_c)
            action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        if self.standard_action:
            return [mu_a, mu_b, mu_c, mu_d], [sigma_a, sigma_b, sigma_c, sigma_d], [action_a[0], action_b[0], action_c[0], action_d[0]]
        else:
            return [mu_a, mu_b, mu_c], [sigma_a, sigma_b, sigma_c], [action_a[0], action_b[0], action_c[0]]

    # deterministic output
    def deterministic_forward(self, x):
        # p(s)
        s = self.fc_input(x)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        _, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s)
        state_sample_a = torch.cat((s, mu_a), dim=1)
        normal_b = self.fc_action_b(state_sample_a)
        _, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, mu_a, mu_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        _, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        if self.standard_action:
            state_sample_a_b_c = torch.cat((s, mu_a, mu_b, mu_c), dim=1)
            normal_d = self.fc_action_d(state_sample_a_b_c)
            _, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # output the mean value to be the deterministic action
        if self.standard_action:
            return mu_a[0][0], mu_b[0][0], mu_c[0][0], mu_d[0][0]
        else:
            return mu_a[0][0], mu_b[0][0], mu_c[0][0]


class REINFORCE(BasePolicy):
    name = 'reinforce'
    type = 'onpolicy'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 20
        self.standard_action = scenario_config['standard_action']
        self.num_scenario = scenario_config['num_scenario']
        self.model = CUDA(AutoregressiveModel(self.standard_action, self.num_waypoint))
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def proceess_init_state(self, state):
        processed_state_list = []
        for s_i in range(len(state)):
            route = state[s_i]['route']
            target_speed = state[s_i]['target_speed']

            index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()
            index = [int(i) for i in index]
            route_norm = normalize_routes(route[index])[:, 0] # [num_waypoint*2]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
            processed_state_list.append(processed_state)
        
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def get_action(self, state, deterministic):
        return [None] * self.num_scenario

    def get_init_action(self, state, deterministic=True):
        # the state should be a sequence of route waypoints
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))

        if deterministic:
            with torch.no_grad():
                if self.standard_action:
                    action_a, action_b, action_c, action_d = self.model.deterministic_forward(processed_state)
                else:
                    action_a, action_b, action_c = self.model.deterministic_forward(processed_state)

            if self.standard_action:
                return [action_a.cpu().numpy(), action_b.cpu().numpy(), action_c.cpu().numpy(), action_d.cpu().numpy()]
            else:
                return [action_a.cpu().numpy(), action_b.cpu().numpy(), action_c.cpu().numpy()]
        else:
            mu_bag, sigma_bag, action_bag = self.model(processed_state)

            # calculate the probability that this distribution outputs this action
            prob_a = normal(action_bag[0], mu_bag[0], sigma_bag[0])
            prob_b = normal(action_bag[1], mu_bag[1], sigma_bag[1])
            prob_c = normal(action_bag[2], mu_bag[2], sigma_bag[2])
            if self.standard_action:
                prob_d = normal(action_bag[3], mu_bag[3], sigma_bag[3])
                log_prob = prob_a.log() + prob_b.log() + prob_c.log() + prob_d.log()
            else:
                log_prob = prob_a.log() + prob_b.log() + prob_c.log()

            # calculate the entropy
            entropy_a = -0.5*((sigma_bag[0]+2*pi.expand_as(sigma_bag[0])).log()+1)
            entropy_b = -0.5*((sigma_bag[1]+2*pi.expand_as(sigma_bag[1])).log()+1)
            entropy_c = -0.5*((sigma_bag[2]+2*pi.expand_as(sigma_bag[2])).log()+1)
            if self.standard_action:
                entropy_d = -0.5*((sigma_bag[2]+2*pi.expand_as(sigma_bag[2])).log()+1)
                entropy = entropy_a + entropy_b + entropy_c + entropy_d
            else:
                entropy = entropy_a + entropy_b + entropy_c

            a_1 = action_bag[0][0].detach().cpu().numpy()
            a_2 = action_bag[1][0].detach().cpu().numpy()
            a_3 = action_bag[2][0].detach().cpu().numpy()
            if self.standard_action:
                a_4 = action_bag[3][0].detach().cpu().numpy()

            if self.standard_action:
                return [a_1, a_2, a_3, a_4], log_prob, entropy
            else:
                return [a_1, a_2, a_3], log_prob, entropy

    def load_model(self):
        if os.path.exists(self.model_path):
            self.logger.log(f'>> Loading LC model from {self.model_path}')
            with open(self.model_path, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to load LC model from {self.model_path}', color='red')

    def save_model(self):
        self.logger.log(f'>> Saving LC model to {self.model_path}')
        with open(self.model_path, 'wb') as f:
            torch.save({'parameters': self.model.state_dict()}, f)
