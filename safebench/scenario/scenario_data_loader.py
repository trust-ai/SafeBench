'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-15 12:40:40
Description: 
'''

import numpy as np


class ScenarioDataLoader:
    def __init__(self, config_lists, num_scenario):
        self.num_scenario = num_scenario
        self.config_lists = config_lists
        self.num_total_scenario = len(config_lists)
        self.reset_idx_counter()

    def reset_idx_counter(self):
        self.scenario_idx = list(range(self.num_total_scenario))

    def _check_overlap(self):
        return 

    def __len__(self):
        return len(self.scenario_idx)

    def sampler(self):
        # sometimes the length of list is smaller than num_scenario
        sample_num = np.min([self.num_scenario, len(self.scenario_idx)])

        # TODO: sampled scenario should not have overlap
        self._check_overlap()

        # select scenarios
        selected_idx = np.random.choice(self.scenario_idx, size=sample_num, replace=False)
        selected_scenario = []
        for s_i in selected_idx:
            selected_scenario.append(self.config_lists[s_i])
            self.scenario_idx.remove(s_i)

        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)
