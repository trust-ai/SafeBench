'''
Author: 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-06 19:41:21
Description: 
'''

import numpy as np

from safebench.agent.safe_rl.policy import DDPG, PPO, SAC, TD3, PPOLagrangian, SACLagrangian, DDPGLagrangian, TD3Lagrangian


# a list of implemented algorithms
POLICY_LIST = {
    "ppo": PPO,
    "ppo_lag": PPOLagrangian,
    "sac": SAC,
    "sac_lag": SACLagrangian,
    "td3": TD3,
    "td3_lag": TD3Lagrangian,
    "ddpg": DDPG,
    "ddpg_lag": DDPGLagrangian,
}

class RLAgent():
    """ 
        Works as an wrapper for all RL agents.
    """
    def __init__(self, config):
        self.agent_name = config['agent_name']
        self.action_dim = config['action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'

        algo_config = config[self.agent_name]
        self.policy = POLICY_LIST[self.agent_name](algo_config)

        if algo_config['policy_type'] == 'on':
            self.train_model = self.train_one_epoch_on_policy
        else:
            self.train_model = self.train_one_epoch_off_policy

    def get_action(self, obs):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        return np.random.randn(batch_size, self.action_dim)

    def load_model(self):
        self.policy.load_model(self.model_path)

    def set_mode(self, mode):
        self.mode = mode

    def train_one_epoch_off_policy(self, buffer, train_steps):
        for _ in range(train_steps):
            data = buffer.get_sample()
            self.policy.learn_on_batch(data)

    def train_one_epoch_on_policy(self, buffer):
        data = buffer.get_sample()
        self.policy.learn_on_batch(data)
