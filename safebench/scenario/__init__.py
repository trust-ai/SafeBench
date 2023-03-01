'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:42:26
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

# collect policy models from scenarios
from safebench.scenario.scenario_policy.dummy_policy import DummyAgent
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE


SCENARIO_POLICY_LIST = {
    'standard': DummyAgent,
    'ordinary': DummyAgent,
    'LC': REINFORCE
}
