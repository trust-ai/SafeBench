'''
@Author: 
@Email: 
@Date: 2020-01-24 13:52:10
LastEditTime: 2023-02-22 23:28:07
@Description: 
'''


class DummyAgent(object):
    """ This agent is used for scenarios that do not have controllable agents. """
    def __init__(self, config, logger):
        self.__name__ = 'dummy'

        self.logger = logger
        self.logger.log('>> This scenario does not require policy model, using a dummy one', color='yellow')
    
    def get_action(self, state):
        return None
