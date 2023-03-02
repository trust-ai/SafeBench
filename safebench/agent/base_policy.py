'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 19:57:03
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    def __init__(self, config, logger):
        pass

    def train(self, replay_buffer):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, deterministic):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def save_model(self):
        raise NotImplementedError()
