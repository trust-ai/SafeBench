from symbol import pass_stmt


import numpy as np


class DummyEgo(object):
    """ This is just an example for testing """
    def __init__(self, config):
        self.action_dim = config['action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'

    def get_action(self, obs):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        return np.random.randn(batch_size, self.action_dim)

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode
