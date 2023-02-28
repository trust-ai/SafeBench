'''
Author: 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-27 20:53:37
Description: 
'''

import numpy as np


class DummyEgo(object):
    name = 'dummy'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.auto_ego = config['auto_ego']
        self.mode = 'train'

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

    def update(self):
        pass
