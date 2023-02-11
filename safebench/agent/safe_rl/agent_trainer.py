import time
from tqdm import tqdm
import traceback

from safebench.agent.safe_rl.worker import OffPolicyWorker, OnPolicyWorker



WORKER_LIST = {
    "ppo": OnPolicyWorker,
    "sac": OffPolicyWorker,
    "td3": OffPolicyWorker,
    "ddpg": OffPolicyWorker,
}


class AgentTrainer:
    def __init__(self, agent_config, logger):
        self.logger = logger
        self.verbose = agent_config['verbose']
        self.epochs = agent_config['epochs']
        self.policy_config = agent_config[agent_config['policy_name']]
        self.worker_config = self.policy_config["worker_config"]
        self.worker_config['ego_action_dim'] = agent_config['ego_action_dim']
        self.worker_config['ego_state_dim'] = agent_config['ego_state_dim']
        self.worker_config['obs_type'] = agent_config['obs_type']
        self.worker = WORKER_LIST[agent_config['policy_name']](self.worker_config, self.logger)
        self.env = None
        self.policy = None

    def set_environment(self, env, agent):
        self.env = env
        self.policy = agent
        self.worker.set_environment(env, agent)

    def train(self):
        continue_from_epoch = self.policy.load_model()
        if continue_from_epoch is None:
            continue_from_epoch = 0
        start_time = time.time()
        total_steps = 0
        for epoch in range(continue_from_epoch, self.epochs):
            epoch_steps = self.worker.train_one_epoch(epoch)
            total_steps += epoch_steps

            for _ in range(self.evaluate_episode_num):
                self.worker.eval()

            if hasattr(self.policy, "post_epoch_process"):
                self.policy.post_epoch_process()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({'env': None}, None)
            # Log info about epoch
            self.data_dict = self._log_metrics(epoch, total_steps, time.time() - start_time, self.verbose)

    def _log_metrics(self, epoch, total_steps, time=None, verbose=True):
        self.logger.log_tabular('CostLimit', self.cost_limit)
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TotalEnvInteracts', total_steps)
        for key in self.logger.logger_keys:
            self.logger.log_tabular(key, average_only=True)
        if time is not None:
            self.logger.log_tabular('Time', time)
        # data_dict contains all the keys except Epoch and TotalEnvInteracts
        data_dict = self.logger.dump_tabular(
            x_axis="TotalEnvInteracts",
            verbose=verbose,
        )
        return data_dict
