'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-05-30 14:20:04
LastEditTime: 2020-09-14 10:45:57
@Description: MDP model should be incorporated with DQN or A2C
'''

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import sys
import copy
import datetime

from loguru import logger
from utils import reward_process, class_from_path, action_normalization_batch, generator_process
from utils import action_process, load_config, select_control_model, save_intersection_image


# 1. ===== import envs =====
import gym
# DO NOT install this envs, otherwise the changes in highway_env will not automatically be valid
sys.path.append('./highway-env')
import highway_env
env = gym.make('intersection-v1')


# 2. ===== parameter settings =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', default=None, type=str, help='select the policy model name. DQN/A2C/DDPG/PPO_GAE/SAC/MBRL')
parser.add_argument('-g', default='none', type=str, help='use which kind of generator [none/uniform/mmrs]')
command_args = parser.parse_args()
policy_model = command_args.m
control_model = select_control_model(policy_model)
args = load_config('./config/intersection_config.yaml')
args_g = args['generator']
args_a = args[policy_model]
args_a['state_dim'] = env.observation_space.shape[0]*env.observation_space.shape[1] # there are two vehicles
#args_a['state_dim'] = 4 # only consider one vehicle
states_g = []
num_route = args_g['state_dim']
for r_i in range(num_route):
    onehot = np.eye(num_route)[[r_i]].astype('float32')
    states_g.append(onehot[0])
route_choose = states_g[0]


# 3. ===== define generator and agent, load generator model =====
from model import Generator
generator = Generator(args_g)
generator.load_model()
sys.path.append('./agents')
agent = class_from_path(args_a['module_path'])(args_a)
agent.load_model()
logger.info('Model name: {}'.format(agent.name))


# 4. ===== prepare test generator =====
if command_args.g == 'none':
    test_episode = 1 # test is deterministic
    test_generator = ['none']
elif command_args.g == 'uniform':
    test_episode = 50
    test_generator = ['uniform', 'mmrs']
elif command_args.g == 'mmrs':
    test_episode = 50
    test_generator = ['uniform']
else:
    raise NotImplementedError()
test_rewards = {}
collision_rates = {}
for t_g in test_generator:
    test_rewards[t_g] = []
    collision_rates[t_g] = []


# 5. ===== start to train =====
mbrl_train_eps = 200
test_interval = 10
episodes = 1050
risk_sigma = 0.2
for e_i in range(episodes):
    '''
    initial_condition = generator_process(command_args.g, generator, route_choose, args_g['action_scale'], risk_sigma)
    env.create_scenario(initial_condition[0], route_choose, ego_model=control_model)
    state_a, ego_goal = env.reset()

    done = False
    # for MBRL method, we can skip the training when model is stable
    if agent.name == 'MBRL' and e_i > mbrl_train_eps:
        done = True

    ep_reward = 0
    while not done:
        #env.render()
        action = agent.select_action(state_a, False)
        next_state_a, reward, done, info = env.step(action_process(action, agent.name))
        post_reward, risk = reward_process(reward, info)
        ep_reward += post_reward

        # off-policy methods
        if agent.name in ['DQN', 'DDPG', 'SAC']:
            agent.store_transition([state_a, action, post_reward, next_state_a, done, risk])
            agent.train()
        # on-policy methods
        elif agent.name in ['A2C', 'PPO_GAE']:
            agent.store_transition(post_reward, state_a, action)
        # mbrl
        elif agent.name == 'MBRL':
            agent.store_transition([state_a, action, next_state_a-state_a, ego_goal])
            #agent.test(state_a, action, next_state_a-state_a)
        state_a = copy.deepcopy(next_state_a)

    # on-policy methods
    if agent.name in ['A2C', 'PPO_GAE']:
        agent.train(state_a)
    # mbrl
    elif agent.name == 'MBRL' and e_i < mbrl_train_eps:
        agent.train()
    #print('[{}/{}] reward: {}'.format(e_i, episodes, ep_reward))
    '''
    
    # start to test
    if (e_i+1) % test_interval == 0:
        for g_i in test_generator: # for each test generator, separately save the results
            test_reward_one_episode = []
            collision_num = 0
            for t_i in range(test_episode):
                initial_condition = generator_process(g_i, generator, route_choose, args_g['action_scale'], risk_sigma)
                env.create_scenario(initial_condition[0], route_choose, ego_model=control_model)
                state_a, ego_goal = env.reset()

                done = False
                test_reward = 0
                frame_id = 0
                while not done:
                    env.render()
                    #image = env.render(mode='rgb_array')
                    #save_intersection_image(image, e_i, frame_id)
                    #frame_id += 1
                    action = agent.select_action(state_a, True)
                    next_state_a, reward, done, info = env.step(action_process(action, agent.name))
                    post_reward, _ = reward_process(reward, info)
                    state_a = copy.deepcopy(next_state_a)
                    test_reward += post_reward
                    if reward['risk_dis'] == 0:
                        collision_num += 1
                print('[{}/{}] Test reward: {}'.format(e_i, episodes, test_reward)) 
                test_reward_one_episode.append(test_reward)
