'''
Author: 
Email: 
Date: 2023-02-04 16:30:08
LastEditTime: 2023-02-26 22:16:32
Description: 
'''

from .dummy_agent.dummy import DummyEgo

from .object_detection.detector import ObjectDetection

from .rl.SAC import SAC
# from .rl.DDPG import DDPG
# from .rl.PPO_GAE import PPO_GAE
# from .rl.MBRL import MBRL


AGENT_POLICY_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
    'sac': SAC,
    #'ddpg': DDPG,
    #'ppo': PPO_GAE,
    #'mbrl': MBRL,
}
