'''
Author: 
Email: 
Date: 2023-01-30 22:30:28
LastEditTime: 2023-02-22 23:28:21
Description: 
'''

# collect policy models from scenarios
from safebench.scenario.srunner.scenarios.policy.dummy_policy import DummyAgent
from safebench.scenario.srunner.scenarios.policy.reinforce_continuous import REINFORCE


SCENARIO_POLICY_LIST = {
    'standard': DummyAgent,
    'LC': REINFORCE
}
