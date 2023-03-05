'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 13:45:51
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np

from safebench.agent.base_policy import BasePolicy
from carla.agents.navigation.behavior_agent import BehaviorAgent 
from carla.agents.navigation.basic_agent import BasicAgent  
from carla.agents.navigation.constant_velocity_agent import ConstantVelocityAgent


class PIDAgent(BasePolicy):
    name = 'pid'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.ego_vehicles = None
        self.route = None

        # define the PID controller
        _dt = 0.05
        _args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': _dt}
        _args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': _dt}

        for e_i in range(config['num_scenario']):
            self.agent_list.append(BasicAgent(self.ego_vehicles[e_i], target_speed=30))

    def set_ego_vehicles(self, ego_vehicles):
        self.ego_vehicles = ego_vehicles

    def set_route(self, route):
        self.route = route

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, deterministic=False):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        action = np.random.randn(batch_size, self.ego_action_dim)
        action[:, 0] = 0.2
        action[:, 1] = 0
        return action

    def load_model(self):
        pass

    def save_model(self):
        pass
