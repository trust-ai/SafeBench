'''
Author: 
Email: 
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-22 23:28:48
Description: 
'''

import carla


class ScenarioTrainer:
    def __init__(self, scenario_config, logger):
        self.logger = logger

    def set_environment(self, env, agent, dataloader):
        if agent.__name__ == 'dummy':
            print('skip training for dummy agent')

    def train(self):
        pass
