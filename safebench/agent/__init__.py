'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 19:57:19
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

from safebench.agent.dummy import DummyAgent
from safebench.agent.object_detection.detector import ObjectDetection
from safebench.agent.rl.sac import SAC


AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'object_detection': ObjectDetection,
    'sac': SAC,
}
