import time

import numpy as np
import carla
import pygame

from safebench.agent import AGENT_LIST
from safebench.gym_carla.env_wrapper import carla_env
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider


class CarlaRunner(object):
    """ Main body to coordinate agents and scenarios. """
    def __init__(self, agent_config, scenario_config):
        self.mode = scenario_config['mode'].lower()
        self.num_scenario = scenario_config['num_scenario']
        self.map_town_config = scenario_config['map_town_config']
        self.obs_type = agent_config['obs_type'] # the observation type is determined by the ego agent
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']

        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        self.frame_skip = scenario_config['frame_skip']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)

        # for visualization
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

    def _init_world(self, town):
        # TODO: before init world, clear all things
        world = self.client.load_world(town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(world)
        world.set_weather(carla.WeatherParameters.ClearNoon)

        print("######## init world completed ########")
        return world

    def _init_renderer(self, num_envs):
        """ Initialize the birdeye view renderer. """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size * num_envs),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }

    def run(self):
        for town in self.map_town_config.keys():
            world = self._init_world(town)
            config_lists = self.map_town_config[town]

            # create and reset scenarios
            env_list = []
            obs_list = []
            for i in range(self.num_scenario):
                config = config_lists[i]
                env = carla_env(self.obs_type, world=world)
                env.create_ego_object()
                raw_obs, ep_reward, ep_len, ep_cost = env.reset(config=config, ego_id=i), 0, 0, 0
                env_list.append(env)
                obs_list.append(raw_obs)

                # load model for scenarios 
                if self.mode in ['eval', 'train_agent'] or self.continue_scenario_training:
                    env.load_model()

            finished_env = set()
            while True: 
                if len(finished_env) == self.num_scenario:
                    print("All scenarios are completed. Prepare for exiting")
                    break
                
                # get action from ego agent (assume using one batch)
                actions_list = self.agent.get_action(obs_list)

                # TODO: move render function to here

                # apply action to env and get obs
                self._update_env(env_list=env_list, obs_list=obs_list, actions_list=actions_list, world=world, finished_env=finished_env)

                # train or test
                if self.mode == 'train_agent':
                    self.agent.add_buffer(obs_list, actions_list)
                    self.agent.train_model()
                    self.agent.save_model()
                elif self.mode == 'train_scenario':
                    for k in range(len(env_list)):
                        env_list[k].add_buffer(obs_list, actions_list)
                        env_list[k].train_model()
                        env_list[k].save_model()

            # TODO: move this display function in the middle of episode
            self._init_renderer(len(env_list))
            self._render_display(env_list, world)

    def _render_display(self, env_list, world):
        birdeye_render_list = []

        max_len = 0
        for cur_env in env_list:
            birdeye_render = BirdeyeRender(world, self.birdeye_params)
            birdeye_render.set_hero(cur_env.ego, cur_env.ego.id)
            birdeye_render_list.append(birdeye_render)
            max_len = max(len(cur_env.render_result), max_len)

        for i in range(max_len):
            for j in range(len(env_list)):
                if i >= len(env_list[j].render_result):
                    continue
                cur_render_result = env_list[j].render_result[i]
                cur_birdeye_render = birdeye_render_list[j]

                cur_birdeye_render.vehicle_polygons = cur_render_result[0]
                cur_birdeye_render.walker_polygons = cur_render_result[1]
                cur_birdeye_render.waypoints = cur_render_result[2]

                self.display.blit(cur_render_result[4], (0, j * self.display_size))
                self.display.blit(cur_render_result[5], (self.display_size, j * self.display_size))
                self.display.blit(cur_render_result[6], (self.display_size * 2, j * self.display_size))
                pygame.display.flip()
            time.sleep(0.1)

    def _update_env(self, env_list, obs_list, actions_list, world, finished_env, render=True):
        reward = [0] * len(env_list)
        cost = [0] * len(env_list)
        info = [None] * len(env_list)
        o = [None] * len(env_list)
        for frame_skip in range(FRAME_SKIP):
            for j in range(len(env_list)):
                env = env_list[j]
                if not env.is_running and env not in finished_env:
                    finished_env.add(env)
                if env in finished_env:
                    continue
                #TODO: seperate step to step_before and step_after
                re_o, re_reward, re_done, re_info, re_cost = env.step(actions_list[j], reward[j], cost[j])
                if re_done:
                    env.is_running = False
                    continue
                reward[j] = re_reward
                cost[j] = re_cost
                info[j] = re_info
                o[j] = re_o

            # tick all scenarios
            world.tick()

        # deal with the rendering results
        for k in range(len(env_list)):
            if env_list[k] in finished_env:
                continue
            if render:
                env_list[k].render()
            ep_reward = obs_list[k][1]
            ep_len = obs_list[k][2]
            ep_cost = obs_list[k][3]
            if "cost" in info[k]:
                ep_cost += info[k]["cost"]
            ep_reward += reward[k]
            ep_len += 1
            obs_list[k] = [o[k], ep_reward, ep_len, ep_cost]
