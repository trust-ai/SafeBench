'''
Author: 
Email: 
Date: 2023-01-30 22:30:20
LastEditTime: 2023-02-26 00:37:49
Description: 
'''

import numpy as np

from safebench.util.run_util import setup_eval_configs
from safebench.agent.safe_rl.policy import DDPG, PPO, SAC, TD3
from safebench.agent.safe_rl.worker import OffPolicyWorker, OnPolicyWorker


# a list of implemented algorithms
POLICY_LIST = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


WORKER_LIST = {
    "ppo": OnPolicyWorker,
    "sac": OffPolicyWorker,
    "td3": OffPolicyWorker,
    "ddpg": OffPolicyWorker,
}


class RLAgent:
    """ 
        Works as an wrapper for all RL agents.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.policy_name = config['policy_name']
        self.ego_action_dim = config['ego_action_dim']
        self.mode = config['mode']

        policy_config = config[self.policy_name]
        policy_config['ego_state_dim'] = config['ego_state_dim']
        policy_config['ego_action_dim'] = config['ego_action_dim']
        policy_config['ego_action_limit'] = config['ego_action_limit']
        self.policy = POLICY_LIST[self.policy_name](policy_config, logger)
        self.load_itr = self.load_model()

    def get_action(self, obs, deterministic=True, with_logprob=False):
        action = []
        for i in range(obs.shape[0]):
            res = self.policy.act(obs[i], deterministic=deterministic, with_logprob=with_logprob)
            action.append(res[0])
        return np.array(action)

    def load_model(self):
        if self.mode in ['eval', 'train_scenario']:
            assert self.config['load_dir'] is not None, "Please specify load_dir!"
        if self.config['load_dir'] is not None:
            model_path, load_itr, _, _, _ = setup_eval_configs(self.config['load_dir'], itr=self.config['load_iteration'])
            self.logger.log(f'>> Loading model from {model_path}')
            self.policy.load_model(model_path)
            return load_itr
        else:
            return None

    # TODO: expose APIs inside policy

    def learn_on_batch(self, data: dict):
        self.policy.learn_on_batch(data)

    def act(self, obs, deterministic=False, with_logprob=False):
        results = []
        for i in range(obs.shape[0]):
            res = self.policy.act(obs[i], deterministic=deterministic, with_logprob=with_logprob)
            results.append(res)
        return_tuple = []
        for i in range(len(results[0])):
            return_tuple.append(np.array([res[i] for res in results]))
        return return_tuple