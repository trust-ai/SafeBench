'''
Author: 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-12 18:06:18
Description: 
'''

import numpy as np


class DummyEgo(object):
    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'

    def get_action(self, obs):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        action = np.random.randn(batch_size, self.ego_action_dim)
        action[:, 0] = 0.2
        action[:, 1] = 0
        return action

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def update(self):
        pass
