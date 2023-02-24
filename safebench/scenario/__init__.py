'''
Author: 
Email: 
Date: 2023-01-30 22:30:28
LastEditTime: 2023-02-24 14:58:28
Description: 
'''

# collect policy models from scenarios
from safebench.scenario.scenario_policy.dummy_policy import DummyAgent
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE


SCENARIO_POLICY_LIST = {
    'standard': DummyAgent,
    'LC': REINFORCE
}
