''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-07 12:28:17
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import copy
import os
import json

import numpy as np
import carla
import pygame
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader, ScenicDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse, scenic_parse

from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores
from safebench.util.scenic_utils import ScenicSimulator

class ScenicRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None
        self.env = None

        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'disable_lidar': True,                                     # show bird-eye view lidar or not
            'display_size': 128,                                       # screen size of one bird-eye view window
            'obs_range': 32,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 12,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 8,                                        # desired speed (m/s)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # define logger
        logger_kwargs = setup_logger_kwargs(self.exp_name, self.output_dir, self.seed,
                                            agent=agent_config['policy_type'],
                                            scenario=scenario_config['policy_type'])
        self.logger = Logger(**logger_kwargs)
        
        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.logger.save_config(agent_config)
        elif self.mode == 'train_scenario':
            self.scene_map = {}
            self.save_freq = scenario_config['save_freq']
            self.logger.create_eval_dir(load_existing_results=False)
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, the action of agent policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        # define agent and scenario policy
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self):
        self.logger.log(">> Initializing carla world")
        self.world = self.client.get_world()
        self.world.scenic = self.scenic
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        
    def _init_scenic(self, config):
        self.logger.log(f">> Initializing scenic simulator: {config.scenic_file}")
        self.scenic = ScenicSimulator(config.scenic_file, self.scenario_config)
        # dummy scene 
        scene, _ = self.scenic.generateScene()
        if self.run_scenes([scene]):
            self.scenic.endSimulation()
        
    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category in ['planning', 'scenic']: 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
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

    def run_scenes(self, scenes):
        self.logger.log(f">> Begin to run the scene...")
        ## currently there is only one scene in this list ##
        for scene in scenes:
            if self.scenic.setSimulation(scene):
                self.scenic.update_behavior = self.scenic.runSimulation()
                next(self.scenic.update_behavior)
                return True
            else:
                return False
        
    def train(self, data_loader, start_episode=0):
        # general buffer for both agent and scenario
        Buffer = RouteReplayBuffer if self.scenario_category in ['planning', 'scenic'] else PerceptionReplayBuffer
        replay_buffer = Buffer(self.num_scenario, self.mode, self.buffer_capacity)

        for e_i in tqdm(range(start_episode, self.train_episode)):
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # TODO: to restart the data loader, reset the index counter every time
            data_loader.reset_idx_counter()

            # get static obs and then reset with init action 
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # start loop
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # apply action to env and get obs
                next_obs, rewards, dones, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)
                obs = copy.deepcopy(next_obs)

                # train on-policy agent or scenario
                if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                    self.agent_policy.train(replay_buffer)

            # end up environment
            self.env.clean_up()
            replay_buffer.finish_one_episode()

            # train off-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)

            # eval during training
            if (e_i+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                self.logger.log('>> ' + '-' * 40)

            # save checkpoints
            if (e_i+1) % self.save_freq == 0:
                self.agent_policy.save_model(e_i)
        
    def eval(self, data_loader, select = False):
        num_finished_scenario = 0
        video_count = 0
        data_loader.reset_idx_counter()
        # recording the score and the id of corresponding selected scenes
            
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario
            assert num_sampled_scenario == 1, 'scenic can only run one scene at one time'
            
            scenes = [config.scene for config in sampled_scenario_configs]
            # begin to run the scene
            self.run_scenes(scenes)
            
            # reset envs with new config, get init action from scenario policy, and run scenario
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # apply action to env and get obs
                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # save video
                if self.save_video:
                    self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    score = rewards[reward_idx] if self.scenario_category in ['planning', 'scenic'] else 1-infos[reward_idx]['iou_loss']
                    score_list[s_i['scenario_id']].append(score)
                    reward_idx += 1

            # clean up all things
            self.logger.log(">> All scenarios are completed. Clearning up all actors")
            self.env.clean_up()

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in sampled_scenario_configs]
                self.logger.save_video(data_ids=data_ids)

            # print score for ranking
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', color='yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), color='yellow')

            # calculate evaluation results
            score_function = get_route_scores if self.scenario_category in ['planning', 'scenic'] else get_perception_scores
            all_running_results = self.logger.add_eval_results(records=self.env.running_results)
            all_scores = score_function(all_running_results)
            self.logger.add_eval_results(scores=all_scores)
            self.logger.print_eval_results()
            if len(self.env.running_results) % self.save_freq == 0:
                self.logger.save_eval_results()
        self.logger.save_eval_results()
        
        if select:
            behavior_name = data_loader.behavior
            # TODO: define new selection mechanism
            self.scene_map[behavior_name] = self.select_adv_scene(all_running_results, score_function, data_loader.select_num)
            self.dump_scene_map()
        self.scenic.destroy()
    
    def select_adv_scene(self, results, score_function, select_num):
        # define your own selection mechanism here
        map_id_score = {}
        for i in results.keys():
            map_id_score[i] = score_function({i:results[i]})['final_score']
        selected_scene_id = sorted([item[0] for item in sorted(map_id_score.items(), key=lambda x:x[1])][:select_num])
        return selected_scene_id

    def run(self):
        # get scenario data of different maps
        config_list = scenic_parse(self.scenario_config, self.logger)
        
        for config in config_list:
            # initialize scenic
            self._init_scenic(config)
            # initialize map and render
            self._init_world()
            self._init_renderer()

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(self.env_params, self.scenario_config, self.world, self.birdeye_render, self.display, self.logger)

            # prepare data loader and buffer
            data_loader = ScenicDataLoader(self.scenic, config, self.num_scenario)

            # run with different modes
            if self.mode == 'eval':
                self.agent_policy.load_model()
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader)
            elif self.mode == 'train_agent':
                start_episode = self.check_continue_training(self.agent_policy)
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode)
            elif self.mode == 'train_scenario':
                self.agent_policy.load_model()
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader, select = True)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def check_continue_training(self, policy):
        # load previous checkpoint
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode
    
    def dump_scene_map(self):
        # load previous checkpoint
        scenic_dir = self.scenario_config['scenic_dir']
        f = open(os.path.join(scenic_dir, f"{scenic_dir.split('/')[-1]}.json"), 'w')
        json_dumps_str = json.dumps(self.scene_map, indent=4)
        print(json_dumps_str, file=f)
        f.close()
    
    def close(self):
        pygame.quit() # close pygame renderer
        if self.env:
            self.env.clean_up()
            
    