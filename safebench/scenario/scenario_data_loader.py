'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:42:18
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
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

    def _select_non_overlap_idx(self, remaining_ids, sample_num):
        selected_idx = []
        current_regions = []
        for s_i in remaining_ids:
            if self.config_lists[s_i].route_region not in current_regions:
                selected_idx.append(s_i)
                if self.config_lists[s_i].route_region != "random":
                    current_regions.append(self.config_lists[s_i].route_region)
            if len(selected_idx) >= sample_num:
                break
        return selected_idx

    def __len__(self):
        return len(self.scenario_idx)

    def sampler(self):
        # sometimes the length of list is smaller than num_scenario
        sample_num = np.min([self.num_scenario, len(self.scenario_idx)])

        # select scenarios
        # selected_idx = np.random.choice(self.scenario_idx, size=sample_num, replace=False)
        selected_idx = self._select_non_overlap_idx(self.scenario_idx, sample_num)
        selected_scenario = []
        for s_i in selected_idx:
            selected_scenario.append(self.config_lists[s_i])
            self.scenario_idx.remove(s_i)

        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)
