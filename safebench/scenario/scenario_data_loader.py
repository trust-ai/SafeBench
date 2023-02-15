'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-15 12:01:32
Description: 
'''

import numpy as np


class ScenarioDataLoader:
    def __init__(self, config_lists, num_scenario, type='eval'):
        self.num_scenario = num_scenario
        self.config_lists = config_lists
        self.num_total_scenario = len(config_lists)

        if type == 'eval':
            self.sampler = self._eval_sampler
        else:
            self.sampler = self._round_sampler

    def _check_overlap(self):
        return 

    def __len__(self):
        return len(self.config_lists)

    def _eval_sampler(self):
        # sometimes the length of list is smaller than num_scenario
        sample_num = np.min([self.num_scenario, len(self.config_lists)])

        # TODO: sampled scenario should not have overlap
        self._check_overlap()

        selected_scenario = []
        for _ in range(sample_num):
            s_i = np.random.randint(0, len(self.config_lists))
            selected_scenario.append(self.config_lists.pop(s_i))

        assert len(selected_scenario) <= self.num_scenario, f"number of scenarios is larger than {self.num_scenario}"
        return selected_scenario, len(selected_scenario)

    def _round_sampler(self):
        raise NotImplementedError()
