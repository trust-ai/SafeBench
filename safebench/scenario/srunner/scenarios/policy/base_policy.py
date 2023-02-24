'''
@Author: 
@Email: 
@Date: 2020-01-24 13:52:10
LastEditTime: 2023-02-24 00:33:46
@Description: 
'''


class BasePolicy:
    """ This is the template for implementing the policy for a scenario. """
    def __init__(self, config, logger):
        self.__name__ = 'base'

    def get_action(self, state):
        raise NotImplementedError()
    
    def get_init_action(self, scenario_config):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()
