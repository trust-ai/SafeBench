'''
@Author: 
@Email: 
@Date: 2020-01-24 13:52:10
LastEditTime: 2023-02-27 20:55:34
@Description: 
'''


class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    """ This is the template for implementing the policy for a scenario. """
    def __init__(self, config, logger):
        pass

    def train(self, replay_buffer):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, deterministic):
        raise NotImplementedError()
    
    def get_init_action(self, scenario_config):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()
