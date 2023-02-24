'''
Author: 
Email: 
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-23 22:59:01
Description: 
'''

import carla


class ScenarioTrainer:
    def __init__(self, scenario_config, logger):
        self.logger = logger

    def set_environment(self, env, agent_policy, scenario_policy, dataloader):
        if scenario_policy.__name__ == 'dummy':
            print('skip training training for dummy policy')

    def train(self):
        pass
