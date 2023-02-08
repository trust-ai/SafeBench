'''
@Author:
@Email:
@Date: 2020-06-19 11:45:14
LastEditTime: 2023-02-08 12:27:03
@Description:
'''

import copy

import numpy as np
import carla
import pygame

from safebench.buffer import Buffer
from safebench.agent import AGENT_LIST
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider


class CarlaRunner(object):
    """ Main body to coordinate agents and scenarios. """
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.mode = scenario_config['mode'].lower()
        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.num_episode = scenario_config['num_episode']
        self.map_town_config = scenario_config['map_town_config']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']

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
        agent_config['action_dim'] = scenario_config['ego_action_dim']
        agent_config['state_dim'] = scenario_config['ego_state_dim']

        # prepare ego agent
        self.agent = AGENT_LIST[agent_config['agent_name']](agent_config)
        if self.mode in ['eval', 'train_scenario'] or self.continue_agent_training:
            self.agent.load_model()
        if self.mode in ['eval', 'train_scenario']:
            self.agent.set_mode('eval')
        else:
            self.agent.set_mode('train')

        # save data during interaction
        self.buffer = Buffer(agent_config, scenario_config)

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

    def run(self):
        for town in self.map_town_config.keys():
            # initialize town
            self._init_world(town)
            # initialize the renderer
            self._init_renderer(self.num_scenario)
            config_lists = self.map_town_config[town]
            assert len(config_lists) >= self.num_scenario, "number of config is less than num_scenario ({} < {})".format(len(config_lists), self.num_scenario)

            # create scenarios within the vectorized wrapper
            env = VectorWrapper(self.agent_config, self.scenario_config, self.world, self.birdeye_render, self.display)
            # load model for scenarios
            if self.mode in ['eval', 'train_agent'] or self.continue_scenario_training:
                env.load_model()

            for e_i in range(self.num_episode):
                # reset envs
                obss = env.reset(config_lists)
                while True:
                    if np.sum(env.finished_env) == self.num_scenario:
                        print("######## All scenarios are completed. Prepare for exiting ########")
                        break

                    # get action from ego agent (assume using one batch)
                    ego_actions = self.agent.get_action(obss)

                    # apply action to env and get obs
                    obss_next, rewards, dones, infos = env.step(ego_actions=ego_actions)

                    # save to buffer
                    self.buffer.add(obss, ego_actions, rewards, dones, infos)
                    obss = copy.deepcopy(obss_next)

                    # for different modes
                    if self.mode == 'train_agent':
                        self.agent.train_model(self.buffer)
                        self.agent.save_model()
                    elif self.mode == 'train_scenario':
                        env.train_model(self.buffer)
                        env.save_model()

                print('[{}/{}] Finish one episode'.format(e_i, self.num_episode))
