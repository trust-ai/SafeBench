'''
Author: 
Email: 
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-28 02:15:55
Description: 
'''

import numpy as np


class ReplayBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """
    
    def __init__(self, num_scenario, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0

        # buffers for step info
        self.reset_buffer()

        # buffers for init info
        self.reset_init_buffer()

    def finish_one_episode(self):
        # get total reward for episode
        for s_i in range(self.num_scenario):
            dones = np.where(self.buffer_dones[s_i] == 1)
            start_ = dones[-2] if len(dones) > 1 else -1
            end_ = dones[-1]
            self.buffer_episode_reward.append(np.sum(self.buffer_rewards[s_i][start_+1:end_+1]))

    def store(self, data_list):
        ego_actions = data_list[0]
        scenario_actions = data_list[1]
        obs = data_list[2]
        next_obs = data_list[3]
        rewards = data_list[4]
        dones = data_list[5]
        infos = data_list[6]
        self.buffer_len += len(infos)

        # separate trajectories according to infos
        for s_i in range(len(infos)):
            sid = infos[s_i]['scenario_id']
            self.buffer_ego_actions[sid].append(ego_actions[s_i])
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i])
            self.buffer_obs[sid].append(obs[s_i])
            self.buffer_next_obs[sid].append(next_obs[s_i])
            self.buffer_rewards[sid].append(rewards[s_i])
            self.buffer_dones[sid].append(dones[s_i])

    def store_init(self, data_list):
        static_obs = data_list[0]
        scenario_init_action = data_list[1]
        self.buffer_init_action.append(scenario_init_action)
        self.init_buffer_len += len(scenario_init_action)

    def sample_init(self, batch_size, shuffle=False):
        buffer_init_action = np.stack(self.buffer_init_action, axis=0)
        buffer_episode_reward = np.stack(self.buffer_episode_reward, axis=0)
        num_trajectory = len(self.buffer_init_action)
        start_idx = np.max([0, num_trajectory - batch_size]) 
        return buffer_init_action[start_idx:], buffer_episode_reward[start_idx:]

    def reset_buffer(self):
        self.buffer_ego_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_scenario_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_next_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_rewards = [[] for _ in range(self.num_scenario)]
        self.buffer_dones = [[] for _ in range(self.num_scenario)]

    def reset_init_buffer(self):
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.init_buffer_len = 0

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_ego_actions = []
        prepared_scenario_actions = []
        prepared_obs = []
        prepared_next_obs = []
        prepared_rewards = []
        prepared_dones = []

        # get the length of each sub-buffer
        samples_per_trajectory = self.buffer_capacity // self.num_scenario # assume average over all sub-buffer
        for s_i in range(self.num_scenario):
            # select the latest samples starting from the end of buffer
            num_trajectory = len(self.buffer_rewards[s_i])
            start_idx = np.max([0, num_trajectory - samples_per_trajectory])

            # concat
            prepared_ego_actions += self.buffer_ego_actions[s_i][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
            prepared_obs += self.buffer_obs[s_i][start_idx:]
            prepared_next_obs += self.buffer_next_obs[s_i][start_idx:]
            prepared_rewards += self.buffer_rewards[s_i][start_idx:]
            prepared_dones += self.buffer_dones[s_i][start_idx:]

        # sample from concatenated list
        sample_index = np.random.randint(0, len(prepared_rewards), size=batch_size)
        if self.mode == 'train_agent':
            action = np.stack(prepared_ego_actions)[sample_index]       # action of agent
        else:
            action = np.stack(prepared_scenario_actions)[sample_index]  # action of scenario
        batch = {
            'action': action,                                         # action
            'state': np.stack(prepared_obs)[sample_index, :],         # state
            'n_state': np.stack(prepared_next_obs)[sample_index, :],  # next state
            'reward': np.stack(prepared_rewards)[sample_index],       # reward
            'done': np.stack(prepared_dones)[sample_index],           # done
        }
        return batch
