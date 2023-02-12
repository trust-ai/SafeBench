import copy

import numpy as np
import carla
import pygame

from safebench.agent import AGENT_LIST
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.agent.safe_rl.agent_trainer import AgentTrainer
from safebench.scenario.srunner.scenario_manager.scenario_trainer import ScenarioTrainer
from safebench.agent.safe_rl.util.logger import EpochLogger, setup_logger_kwargs


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.mode = scenario_config['mode']
        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.num_episode = scenario_config['num_episode']
        self.map_town_config = scenario_config['map_town_config']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_type = scenario_config['type_name'].split('.')[0]

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None

        # for obtaining rendering results
        self.display_size = 256
        self.obs_range = 32
        self.d_behind = 12

        # pass info from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # prepare ego agent
        if self.mode == 'eval':
            self.logger = EpochLogger(eval_mode=True)
        elif self.mode == 'train_scenario':
            self.logger = EpochLogger(eval_mode=True)
            self.trainer = ScenarioTrainer()
        elif self.mode == 'train_agent':
            logger_kwargs = setup_logger_kwargs(scenario_config['exp_name'], scenario_config['seed'], data_dir=scenario_config['data_dir'])
            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(agent_config)
            self.trainer = AgentTrainer(agent_config, self.logger)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")
        self.agent = AGENT_LIST[agent_config['agent_type']](agent_config, logger=self.logger)
        self.env = None

    def _init_world(self, town):
        print("######## initializeing carla world ########")
        # TODO: before init world, clear all things
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self, num_envs):
        print("######## initializeing pygame birdeye renderer ########")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        self.display = pygame.display.set_mode((self.display_size * 3, self.display_size * num_envs), flag)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }

        # initialize the render for genrating observation and visualization
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params)

    def eval(self):
        for e_i in range(self.num_episode):
            # reset envs
            obss = self.env.reset()
            rewards_list = {e_i: [] for e_i in range(self.num_scenario)}
            while True:
                if np.sum(self.env.finished_env) == self.num_scenario:
                    print("######## All scenarios are completed. Prepare for exiting ########")
                    break

                # get action from ego agent (assume using one batch)
                ego_actions = self.agent.get_action(obss)

                # apply action to env and get obs
                obss_next, rewards, dones, infos = self.env.step(ego_actions=ego_actions)

                # accumulate reward to corresponding scenario
                reward_idx = 0
                for e_i in self.num_scenario:
                    if self.env.finished_env[e_i]:
                        rewards_list[e_i].append(reward_idx)
                        reward_idx += 1
            
            # calculate episode reward and print
            print('[{}/{}] Episode reward for {} scenarios:'.format(e_i, self.num_episode, self.num_scenario))
            for e_i in rewards_list.keys():
                print('\t Scenario', e_i, '-', np.sum(rewards_list[e_i]))

    def run(self):
        for town in self.map_town_config.keys():
            # initialize town
            self._init_world(town)
            # initialize the renderer
            self._init_renderer(self.num_scenario)
            config_lists = self.map_town_config[town]
            assert len(config_lists) >= self.num_scenario, "number of config is less than num_scenario ({} < {})".format(len(config_lists), self.num_scenario)

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(self.agent_config, self.scenario_config, self.world, self.birdeye_render, self.display, config_lists, self.scenario_type)

            if self.mode == 'eval':
                self.eval()
            elif self.mode in ['train_scenario', 'train_agent']:
                self.trainer.set_environment(self.env, self.agent)
                self.trainer.train()
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")
