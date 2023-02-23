'''
Author: 
Email: 
Date: 2023-02-04 16:30:08
LastEditTime: 2023-02-22 19:59:28
Description: 
'''

from .dummy_agent.dummy import DummyEgo
from .object_detection.detector import ObjectDetection
from .safe_rl.rl_agent import RLAgent


AGENT_POLICY_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
    'rl': RLAgent
}
