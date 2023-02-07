'''
@Author: 
@Email: 
@Date: 2020-06-19 11:45:14
LastEditTime: 2023-02-06 20:38:39
@Description: 
'''

import numpy as np
import torch

from safebench.util.torch_util import to_tensor


# TODO: if agent and scenario share the same buffer, we should use two index list to seperate the buffer to avoid duplicated storing
# TODO: data from different scenarios should be stored seperately in one episode
class Buffer:
    def __init__(self, agent_config, scenario_config):
        self.obs_dim = agent_config['state_dim'] 
        self.act_dim = agent_config['action_dim'] 
        self.max_buffer_size = agent_config['max_buffer_size'] 
        self.num_scenario = scenario_config['num_scenario']
        self.clear()

    def clear(self):
        self.obs_buf = np.zeros((self.max_buffer_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_buffer_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.max_buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_buffer_size, dtype=np.float32)
        self.ptr, self.path_start_idx = 0, 0

    def add(self, obs, act, rew, done, infos):
        """
            We assume the dimension of all data is 2 since we collect data from multuple scenarios
        """
        curr_ptr = self.ptr % self.max_buffer_size, ""
        batch_size = len(obs)

        # self.obs_buf[curr_ptr:curr_ptr+batch_size] = obs
        # self.act_buf[curr_ptr:curr_ptr+batch_size] = act
        # self.rew_buf[curr_ptr:curr_ptr+batch_size] = rew
        # self.done_buf[curr_ptr:curr_ptr+batch_size] = done
        # self.ptr += batch_size

    def finish_path(self):
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr > 0  # buffer has to have something before you can get
        data = dict(
            obs=self.obs_buf[:self.ptr],
            act=self.act_buf[:self.ptr],
            ret=self.ret_buf[:self.ptr],
            done=self.done_buf[:self.ptr],
        )
        tensor_dict = to_tensor(data, dtype=torch.float32)
        return tensor_dict
