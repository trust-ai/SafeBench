'''
Author: 
Email: 
Date: 2023-02-04 16:30:08
LastEditTime: 2023-02-06 19:26:04
Description: 
'''

from .dummy_agent.dummy import DummyEgo
from .object_detection.detector import ObjectDetection
from .safe_rl.rl_agent import RLAgent


AGENT_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
    'rl': RLAgent
}
