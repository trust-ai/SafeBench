'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 17:10:10
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np

from safebench.agent.base_policy import BasePolicy
from agents.navigation.basic_agent import BasicAgent  


class CarlaBasicAgent(BasePolicy):
    name = 'basic'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        self.target_speed = 30

        self.opt_dict = {
            #'lateral_control_dict': {'K_P': 1.0, 'K_I': 0.0, 'K_D': 0.0, 'dt': 0.5},
            #'longitudinal_control_dict': {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.5},
            #'max_steering': 0.8,
        }

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(self.num_scenario):
            controller = BasicAgent(self.ego_vehicles[e_i], target_speed=self.target_speed, opt_dict=self.opt_dict)
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # set route for each controller
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step()
            throttle = control.throttle
            steer = control.steer
            actions.append([throttle, -steer]) # TODO: consistent with gym-carla
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
