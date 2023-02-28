'''
Author: 
Email: 
Date: 2023-02-04 16:30:08
LastEditTime: 2023-02-27 20:44:24
Description: 
'''

from safebench.agent.dummy import DummyEgo
from safebench.agent.object_detection.detector import ObjectDetection
from safebench.agent.rl.sac import SAC
# from safebench.agent.rl.DDPG import DDPG
# from safebench.agent.rl.PPO_GAE import PPO_GAE


AGENT_POLICY_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
    'sac': SAC,
    #'ddpg': DDPG,
    #'ppo': PPO_GAE,
    #'mbrl': MBRL,
}
