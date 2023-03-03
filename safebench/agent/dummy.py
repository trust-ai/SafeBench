'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 19:55:00
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np


class DummyAgent(object):
    name = 'dummy'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.auto_ego = config['auto_ego']
        self.mode = 'train'
        self.continue_episode = 0

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, deterministic=False):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        action = np.random.randn(batch_size, self.ego_action_dim)
        action[:, 0] = None if self.auto_ego else 0.2
        action[:, 1] = None if self.auto_ego else 0
        return action

    def load_model(self):
        pass

    def save_model(self):
        pass
