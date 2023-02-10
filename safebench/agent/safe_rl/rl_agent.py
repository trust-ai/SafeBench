'''
Author: 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-06 19:41:21
Description: 
'''

import numpy as np

from safebench.util.run_util import setup_eval_configs
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
    def __init__(self, config, logger):
        self.policy_name = config['policy_name']
        self.ego_action_dim = config['ego_action_dim']
        self.mode = config['mode']

        policy_config = config[self.policy_name]
        self.policy = POLICY_LIST[self.policy_name](logger, policy_config)
        self.load_model()

        if policy_config['policy_type'] == 'on':
            self.train_model = self.train_one_epoch_on_policy
        else:
            self.train_model = self.train_one_epoch_off_policy


    def get_action(self, obs):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        return np.random.randn(batch_size, self.ego_action_dim)

    def load_model(self):
        if self.mode in ['eval', 'train_scenario']:
            assert config['load_dir'] is not None, "Please specify load_dir!"
        if config['load_dir'] is not None:
            model_path, _, _, _ = setup_eval_configs(config['load_dir'], itr=config['load_iteration'])
            self.policy.load_model(model_path)

    def train_one_epoch_off_policy(self, buffer, train_steps):
        for _ in range(train_steps):
            data = buffer.get_sample()
            self.policy.learn_on_batch(data)

    def train_one_epoch_on_policy(self, buffer):
        data = buffer.get_sample()
        self.policy.learn_on_batch(data)
