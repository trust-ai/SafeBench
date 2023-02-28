'''
Author: 
Email: 
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-28 02:15:24
Description: 
'''

import copy

import numpy as np
import carla
import pygame
import os
import joblib
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import ReplayBuffer

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse

from safebench.util.logger import EpochLogger, setup_logger_kwargs
from safebench.util.run_util import save_video


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.output_dir = scenario_config['output_dir']
        self.save_video = scenario_config['save_video']
        self.mode = scenario_config['mode']
        assert not self.save_video or (self.save_video and self.mode == 'eval'), "only allowed saving video in eval mode"

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['type_category']
        self.scenario_policy_type = scenario_config['type_name'].split('.')[0]

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None

        self.env_params = {
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'disable_lidar': True,
            'display_size': 128,                    # screen size of one bird-eye view windowd=
            'obs_range': 32,                        # observation range (meter)
            'd_behind': 12,                         # distance behind the ego vehicle (meter)
            'max_past_step': 1,                     # the number of past steps to draw
            'discrete': False,                      # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],       # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],     # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'max_episode_step': 100,                # maximum timesteps per episode
            'max_waypt': 12,                        # maximum number of waypoints
            'lidar_bin': 0.125,                     # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                    # threshold for out of lane (meter)
            'desired_speed': 8,                     # desired speed (m/s)
            'image_sz': 1024,                       # TODO: move to config of od scenario
        }

        # pass info from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # define logger
        logger_kwargs = setup_logger_kwargs(scenario_config['exp_name'], scenario_config['seed'])
        self.logger = EpochLogger(**logger_kwargs)
        
        # prepare parameters
        if self.mode == 'eval':
            self.logger = EpochLogger(eval_mode=True)
        elif self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.logger.save_config(agent_config)
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + self.scenario_policy_type)
        self.logger.log('-' * 40)
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[self.scenario_policy_type](scenario_config, logger=self.logger)

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self, num_envs):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category == 'planning': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * num_envs)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * num_envs)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * num_envs)
        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, env, data_loader):
        # general buffer for both agent and scenario
        replay_buffer = ReplayBuffer(self.num_scenario, self.mode, self.buffer_capacity)

        for e_i in tqdm(range(self.train_episode)):
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # TODO: to restart the data loader, reset the index counter every time
            data_loader.reset_idx_counter()

            # get static obs and then reset with init action 
            static_obs = env.get_static_obs(sampled_scenario_configs)
            scenario_init_action = self.scenario_policy.get_init_action(static_obs)
            obs = env.reset(sampled_scenario_configs, scenario_init_action)
            replay_buffer.store_init([static_obs, scenario_init_action])

            # start loop
            while not env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, deterministic=False)

                # apply action to env and get obs
                next_obs, rewards, dones, infos = env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones, infos])
                obs = copy.deepcopy(next_obs)

                # train on-policy agent or scenario
                if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                    self.agent_policy.train(replay_buffer)
                elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                    self.scenario_policy.train(replay_buffer)

            # end up environment
            env.clean_up()
            replay_buffer.finish_one_episode()
            
            # train off-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                self.scenario_policy.train(replay_buffer)

            # eval during training
            if (e_i+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                self.logger.log('>> ' + '-' * 40)

            # save checkpoints
            if (e_i+1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model()
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model()

    def eval(self, env, data_loader):
        num_finished_scenario = 0
        video_count = 0
        data_loader.reset_idx_counter()
        result_dir = os.path.join(self.output_dir, 'eval_results')
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(self.output_dir, 'eval_results/results.pkl')
        if os.path.exists(result_file):
            print('loading previous evaluation results from', result_file)
            eval_results = joblib.load(result_file)
        else:
            eval_results = {}
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = env.get_static_obs(sampled_scenario_configs)
            scenario_init_action = self.scenario_policy.get_init_action(static_obs)
            obs = env.reset(sampled_scenario_configs, scenario_init_action)

            rewards_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            frame_list = []
            while not env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, deterministic=True)

                # apply action to env and get obs
                obs, rewards, _, infos = env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # save video
                if self.save_video:
                    frame_list.append(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))

                # accumulate reward to corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    rewards_list[s_i['scenario_id']].append(rewards[reward_idx])
                    reward_idx += 1

            eval_results.update(env.running_results)

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            env.clean_up()

            self.logger.log('>> Saving evaluation results')
            joblib.dump(eval_results, result_file)

            # save video
            if self.save_video:
                self.logger.log('>> Saving video')
                save_video(frame_list, os.path.join(self.output_dir, 'video/video_{}.gif'.format(str(video_count))))
                video_count += 1

            # calculate episode reward and print
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Episode reward for batch scenario:', color='yellow')
            for s_i in rewards_list.keys():
                self.logger.log('\t Scenario ' + str(s_i) + ': ' + str(np.sum(rewards_list[s_i])), color='yellow')

    def run(self):
        # get scenario data of different maps
        maps_data = scenario_parse(self.scenario_config, self.logger)
        for town in maps_data.keys():
            # initialize town
            self._init_world(town)

            # initialize renderer
            self._init_renderer(self.num_scenario)

            # create scenarios within the vectorized wrapper
            env = VectorWrapper(self.env_params, self.scenario_config, self.world, self.birdeye_render, self.display, self.logger)

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(maps_data[town], self.num_scenario)

            # run with different modes
            if self.mode == 'eval':
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.load_model()
                self.scenario_policy.set_mode('eval')
                self.eval(env, data_loader)
            elif self.mode == 'train_agent':
                self.agent_policy.set_mode('train')
                self.scenario_policy.load_model()
                self.scenario_policy.set_mode('eval')
                self.train(env, data_loader)
            elif self.mode == 'train_scenario':
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(env, data_loader)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def close(self):
        # check if all actors are cleaned
        actor_filters = [
            'sensor.other.collision', 
            'sensor.lidar.ray_cast',
            'sensor.camera.rgb',
            'controller.ai.walker',
            'vehicle.*',
            'walker.*',
        ]
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                self.logger.log('>> Removing agent: ' + str(actor.type_id) + '-' + str(actor.id))
                if actor.is_alive:
                    if actor.type_id.split('.')[0] in ['controller', 'sensor']:
                        actor.stop()
                    actor.destroy()
