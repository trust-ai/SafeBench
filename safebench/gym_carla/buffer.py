import numpy as np


class ReplayBuffer:
    def __init__(self, num_trajectory):
        self.replay_buffer = [[] for _ in range(num_trajectory)]
        self.init_obs = [None for _ in range(num_trajectory)]
        self.current_action = [None for _ in range(num_trajectory)]

    def save_init_obs(self, trajectory_id, obs):
        self.init_obs[trajectory_id] = obs

    def save_current_action(self, trajectory_id, action):
        self.current_action[trajectory_id] = action

    def save_step_results(self, trajectory_id, next_obs, reward, done, info, critic_value=None, log_prob=None):
        if len(self.replay_buffer[trajectory_id]) == 0:
            obs = self.init_obs[trajectory_id]
        else:
            obs = self.replay_buffer[trajectory_id][-1]['obs']
        data = {
            'obs': obs,
            'act': self.current_action[trajectory_id],
            'obs2': next_obs,
            'rew': reward,
            'done': done,
            'info': info,
        }
        if critic_value is not None:
            data['critic_value'] = critic_value
        if log_prob is not None:
            data['log_prob'] = log_prob
        self.replay_buffer[trajectory_id].append(data)

    def get_trajectories(self):
        output = [trajectory for trajectory in self.replay_buffer if len(trajectory) > 0]
        return output
