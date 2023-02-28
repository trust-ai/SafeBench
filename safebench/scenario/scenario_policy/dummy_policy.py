'''
@Author: 
@Email: 
@Date: 2020-01-24 13:52:10
LastEditTime: 2023-02-27 20:55:16
@Description: 
'''

from safebench.scenario.scenario_policy.base_policy import BasePolicy


class DummyAgent(BasePolicy):
    name = 'dummy'
    type = 'unlearnable'

    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        self.logger = logger
        self.logger.log('>> This scenario does not require policy model, using a dummy one', color='yellow')
        self.num_scenario = config['num_scenario']

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, deterministic):
        return [None] * self.num_scenario

    def get_init_action(self, scenario_config):
        return [None] * self.num_scenario

    def load_model(self):
        return None
