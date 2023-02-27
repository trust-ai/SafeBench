'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
LastEditTime: 2020-09-14 23:09:19
@Description: 
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable

import importlib
import os
import yaml


def CPU(var):
    return var.detach().cpu().numpy()


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def normal(x, mu, sigma_sq):
    pi = CUDA(Variable(torch.FloatTensor([np.pi])))
    a = (-1*(CUDA(Variable(x))-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


def action_normalization_batch(action, action_scale):
    action_1 = action[:, 0:1]*action_scale['x']
    action_2 = action[:, 1:2]*action_scale['y']
    action_3 = action[:, 2:3]*action_scale['vx']
    action_4 = action[:, 3:4]*action_scale['vy']

    action_out = np.concatenate([action_1, action_2, action_3, action_4], axis=1)
    return action_out


def action_normalization(action, action_range, action_scale, env_name='intersection'):
    if env_name == 'intersection':
        action_1 = action[0]*action_scale['x']
        action_2 = action[1]*action_scale['y']
        action_3 = action[2]*action_scale['vx']
        action_4 = action[3]*action_scale['vy']

        valid = True
        if np.abs(action_1) > action_range['x']:
            valid = False
        if np.abs(action_2) > action_range['y']:
            valid = False
        if np.abs(action_3) < action_range['vx_l'] or np.abs(action_3) > action_range['vx_h']:
            valid = False
        if np.abs(action_4) < action_range['vy_l'] or np.abs(action_4) > action_range['vx_h']:
            valid = False
        return [action_1, action_2, action_3, action_4], valid
    elif env_name == 'leftturn':
        min_range = action_range[0]
        max_range = action_range[1]
        scale = action_scale[0]
        bias = action_scale[1]
        valid = True
        action_scaled_list = []
        for v_i in range(len(action)):
            action_scaled = action[v_i]*scale[v_i]+bias[v_i]
            action_scaled_list.append(action_scaled)
            if action_scaled < min_range[v_i] or action_scaled > max_range[v_i]:
                valid = False
        return action_scaled_list, valid


def save_realnvp_image(z, episode_id, s_i):
    plt.figure(figsize=(5, 5))
    plt.plot(z[:,0], -z[:, 1], 'b.', zorder=1, alpha=0.1)  # the positive axis of y is down

    route = np.load('./log/trajectories_'+str(s_i)+'.npy')
    route_color = '#ffa500' # 1E90FF
    x = route[:, 0]
    y = -route[:, 1]  # the positive axis of y is down

    draw_x = []
    draw_y = []
    for l_j in range(len(x)):
        draw_x.append(x[l_j])
        draw_y.append(y[l_j])
    plt.plot(draw_x, draw_y, route_color, zorder=10)
    plt.scatter(draw_x, draw_y, s=25, c=route_color, zorder=10)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.tight_layout()
    # s_i is the index of state
    plt.savefig('./tools/gif/'+str(episode_id)+'_'+str(s_i)+'.png', dpi=200, format='png')


def save_intersection_image(img, episode_id, frame_id):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    folder_name = './tools/intersection/'+str(episode_id)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(folder_name+'/'+str(frame_id)+'.png', dpi=200, format='png')
    plt.close('all')


def generate_highway_data(regions, action_scale, number=1000):
    # use uniform distribution to restrict the region that cyclist appears
    xy_list = []
    for r_i in regions:
        x = np.random.uniform(r_i[0], r_i[1], size=number)/action_scale['x']
        y = np.random.uniform(r_i[2], r_i[3], size=number)/action_scale['y']
        vx = np.random.uniform(r_i[4], r_i[5], size=number)/action_scale['vx']
        vy = np.random.uniform(r_i[6], r_i[7], size=number)/action_scale['vy']
        xy = np.concatenate((x[:, None], y[:, None], vx[:, None], vy[:, None]), axis=1).astype('float32')
        xy_list.append(xy)
    return np.concatenate([i for i in xy_list], axis=0)


def random_scenario_generator(region):
    x = np.random.uniform(-region['x'], region['x'])
    y = np.random.uniform(-region['y'], region['y'])
    vx = np.random.uniform(region['vx_l'], region['vx_h'])*np.random.choice([1.0, -1.0], 1)[0]
    vy = np.random.uniform(region['vy_l'], region['vy_h'])*np.random.choice([1.0, -1.0], 1)[0]
    return [x, y, vx, vy]


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def select_control_model(name):
    # IDM and PID models do not need to be train
    if name in ['DQN', 'A2C']:
        return 'MDP'
    elif name in ['DDPG', 'PPO_GAE', 'SAC', 'MBRL']:
        return 'LC'
    else:
        raise ValueError('No policy model matched')


def action_process(action, name):
    mdp_action = ['SLOWER', 'FASTER']
    if name in ['DQN', 'A2C']:
        return mdp_action[action]
    elif name in ['DDPG', 'PPO_GAE', 'SAC', 'MBRL']:
        return action


def generator_process(name, generator, route_choose, action_scale, risk_sigma):
    if name == 'mmrs':
        # generate one scenario from trained generator
        initial_condition = generator.flow_sample(state=route_choose, sample_number=1, sigma=risk_sigma)
        initial_condition = action_normalization_batch(initial_condition, action_scale)
    elif name == 'uniform':
        initial_condition = np.random.uniform(-1.0, 1.0, 4)[None]
        initial_condition = action_normalization_batch(initial_condition, action_scale)
    elif name == 'none':
        # use a far away and static initial condition
        initial_condition = [None]
    elif name == 'single':
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    
    return initial_condition


def reward_process(reward, info):
    risk_reward = reward['risk_dis'] # the risk reward has not been exp
    goal_reward = reward['goal_reward']

    risk = False
    final_reward = goal_reward
    if risk_reward <= 3: # when two vehicles are too close or crashed
        final_reward -= 5
        risk = True

    # road constraint
    # why put it here? because if we use collision checking in highway-env, the cyclist will be blocked 
    road_blocks = [
        [-5, -12, np.pi/2, 15, 0.5], # check left
        [4, 0, np.pi/2, 40, 0.5],    # check right
        [-16, -4.5, np.pi, 22, 0.5], # check up
        [-16, 5, np.pi, 22, 0.5],    # check down
        [-5, 13, np.pi/2, 16, 0.5],  # check left
    ]
    # NOTE: These three regions are not drivable, stop the env when violate the road constraint
    left_top_region = info['pos'][0] < road_blocks[0][0] and info['pos'][1] < road_blocks[2][1]
    left_bottom_region = info['pos'][0] < road_blocks[0][0] and info['pos'][1] > road_blocks[3][1]
    right_region = info['pos'][0] > road_blocks[1][0]
    if left_top_region or left_bottom_region or right_region:
        final_reward -= 5

    '''
    # the direction of velocity should be constrained, vehicle should not go back
    h = info['heading'] # rad
    vx_sign = np.cos(h)*info['vel'][0]
    vy_sign = np.sin(h)*info['vel'][1]
    if vx_sign < 0 and vy_sign < 0:
        print('Reversing !')
        final_reward -= 50
    '''

    '''
    # boundary penalty
    if info['boundary_penaly']:
        final_reward -= 1000
    '''

    return final_reward, risk
