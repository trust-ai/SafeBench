import time
from copy import deepcopy
import os.path as osp
import gym
import gym_carla
import rospy

import torch
from tqdm import tqdm
import traceback

from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus

import sys
sys.path.append('/home/carla/Evaluation/src/agent/gym_node/src/planning')

from planning.safe_rl.policy import DDPG, PPO, SAC, TD3, PPOLagrangian, SACLagrangian, DDPGLagrangian, TD3Lagrangian
from planning.safe_rl.policy.bev_policy import PPO_BEV, DDPG_BEV, SAC_BEV, TD3_BEV
from planning.safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from planning.safe_rl.util.run_util import load_config, setup_eval_configs
from planning.safe_rl.util.torch_util import export_device_env_variable, seed_torch
from planning.safe_rl.worker import OffPolicyWorker, OnPolicyWorker
from planning.env_wrapper import carla_env

class CarlaRunner:
    '''
    Main entry that coodrinate learner and worker
    '''
    # First element is the policy class while the second is whether it is an on-policy algorithm
    POLICY_LIB = {
        "ppo": (PPO, True, OnPolicyWorker),
        "ppo_lag": (PPOLagrangian, True, OnPolicyWorker),
        "sac": (SAC, False, OffPolicyWorker),
        "sac_lag": (SACLagrangian, False, OffPolicyWorker),
        "td3": (TD3, False, OffPolicyWorker),
        "td3_lag": (TD3Lagrangian, False, OffPolicyWorker),
        "ddpg": (DDPG, False, OffPolicyWorker),
        "ddpg_lag": (DDPGLagrangian, False, OffPolicyWorker),
    }

    BEV_POLICY_LIB = {
        "ppo": (PPO_BEV, True, OnPolicyWorker),
        "sac": (SAC_BEV, False, OffPolicyWorker),
        "td3": (TD3_BEV, False, OffPolicyWorker),
        "ddpg": (DDPG_BEV, False, OffPolicyWorker),
    }

    def __init__(self,
                 sample_episode_num=50,
                 episode_rerun_num=10,
                 evaluate_episode_num=1,
                 mode="train",
                 exp_name="exp",
                 seed=0,
                 device="cpu",
                 device_id=0,
                 threads=2,
                 policy="ddpg",
                 timeout_steps=200,
                 epochs=10,
                 save_freq=20,
                 load_dir=None,
                 data_dir=None,
                 record_dir=None,
                 verbose=True,
                 continue_from_epoch=0,
                 obs_type=0,
                 port=2000,
                 role_name=None,
                 **kwarg) -> None:
        seed_torch(seed)
        torch.set_num_threads(threads)
        export_device_env_variable(device, id=device_id)

        self.episode_rerun_num = episode_rerun_num
        self.sample_episode_num = sample_episode_num
        self.evaluate_episode_num = evaluate_episode_num
        self.continue_from_epoch = continue_from_epoch
        self.obs_type = obs_type
        load_dir = load_dir.split("/data/")[-1]
        load_dir = osp.join(osp.dirname(osp.abspath(__file__)) + '/data', load_dir)
        self.load_dir = load_dir

        # Instantiate environment
        env_args = {'port': port, 'role_name': role_name, 'record_dir': record_dir}
        self.env = carla_env(env_args, obs_type)
        self.env.seed(seed)

        mode = mode.lower()
        if mode == "eval":
            # Read some basic env and model info from the dir configs
            assert load_dir is not None, "The load_path parameter has not been specified!!!"
            model_path, policy, timeout_steps, policy_config = setup_eval_configs(
                load_dir)
            self._eval_mode_init(model_path, policy, timeout_steps, policy_config)
        else:
            self._train_mode_init(seed, exp_name, policy, timeout_steps, data_dir,
                                  **kwarg)
            self.batch_size = self.worker_config[
                "batch_size"] if "batch_size" in self.worker_config else None


        self.epochs = epochs
        self.save_freq = save_freq
        self.data_dict = []
        self.epoch = self.continue_from_epoch
        self.verbose = verbose
        if mode == "train" and "cost_limit" in self.policy_config:
            self.cost_limit = self.policy_config["cost_limit"]
        else:
            self.cost_limit = 1e3        

    def _train_mode_init(self, seed, exp_name, policy, timeout_steps, data_dir,
                         **kwarg):

        self.timeout_steps = self.env._max_episode_steps if timeout_steps == -1 else timeout_steps
        config = locals()
        # record some local attributes from the child classes
        attrs = {}
        for k, v in self.__dict__.items():
            if k != "env":
                attrs[k] = deepcopy(v)

        config.update(attrs)
        # remove some non-useful keys
        [config.pop(key) for key in ["self", "kwarg"]]
        config[policy] = kwarg[policy]

        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=data_dir)
        self.logger = EpochLogger(**logger_kwargs)

        self.logger.save_config(config)


        # Init policy
        self.policy_config = kwarg[policy]
        self.policy_config["timeout_steps"] = self.timeout_steps
        self.policy_config["obs_type"] = self.obs_type
        if self.obs_type < 2:
            policy_cls, self.on_policy, worker_cls = self.POLICY_LIB[policy.lower()]
        else:
            policy_cls, self.on_policy, worker_cls = self.BEV_POLICY_LIB[policy.lower()]
        self.policy = policy_cls(self.env, self.logger, **self.policy_config)

        if self.load_dir is not None:
            model_path, _, _, _ = setup_eval_configs(self.load_dir)
            self.policy.load_model(model_path)

        self.steps_per_epoch = self.policy_config[
            "steps_per_epoch"] if "steps_per_epoch" in self.policy_config else 1
        self.worker_config = self.policy_config["worker_config"]
        self.worker = worker_cls(self.env,
                                 self.policy,
                                 self.logger,
                                 timeout_steps=self.timeout_steps,
                                 **self.worker_config)

    def _eval_mode_init(self, model_path, policy, timeout_steps, policy_config):
        self.timeout_steps = self.env._max_episode_steps if timeout_steps == -1 else timeout_steps

        # Set up logger but don't save anything
        self.logger = EpochLogger(eval_mode=True)

        # Init policy
        policy_config["timeout_steps"] = self.timeout_steps
        policy_config["obs_type"] = self.obs_type

        if self.obs_type < 2:
            policy_cls, self.on_policy, worker_cls = self.POLICY_LIB[policy.lower()]
        else:
            policy_cls, self.on_policy, worker_cls = self.BEV_POLICY_LIB[policy.lower()]
        self.policy = policy_cls(self.env, self.logger, **policy_config)

        self.policy.load_model(model_path)

    # @profile
    def train_one_epoch_off_policy(self, epoch):
        epoch_steps = 0
        range_instance = tqdm(
            range(self.sample_episode_num),
            desc='Collecting trajectories') if self.verbose else range(
                self.sample_episode_num)
        for i in range_instance:
            steps = self.worker.work()
            epoch_steps += steps

        train_steps = self.episode_rerun_num * epoch_steps // self.batch_size
        range_instance = tqdm(
            range(train_steps), desc='training {}/{}'.format(
                epoch + 1, self.epochs)) if self.verbose else range(train_steps)
        for i in range_instance:
            data = self.worker.get_sample()
            self.policy.learn_on_batch(data)

        return epoch_steps

    # @profile
    def train_one_epoch_on_policy(self, epoch):
        epoch_steps = 0
        steps = self.worker.work()
        epoch_steps += steps
        data = self.worker.get_sample()
        self.policy.learn_on_batch(data)
        return epoch_steps

    def train(self):
        start_time = time.time()
        total_steps = 0
        for epoch in range(self.continue_from_epoch, self.epochs):
            self.epoch += 1
            if self.on_policy:
                epoch_steps = self.train_one_epoch_on_policy(epoch)
            else:
                epoch_steps = self.train_one_epoch_off_policy(epoch)
            total_steps += epoch_steps

            for _ in range(self.evaluate_episode_num):
                self.worker.eval()

            if hasattr(self.policy, "post_epoch_process"):
                self.policy.post_epoch_process()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({'env': None}, None)
            # Log info about epoch
            self.data_dict = self._log_metrics(epoch, total_steps,
                                               time.time() - start_time, self.verbose)

    def eval(self, sleep=0.01, render=True):
        total_steps = 0
        for epoch in range(self.epochs):
            try:
                raw_obs, ep_reward, ep_len, ep_cost = self.env.wait_for_reset(), 0, 0, 0
                if render:
                    self.env.render()
                for i in range(self.timeout_steps):
                    if self.obs_type > 1:
                        obs = self.policy.process_img(raw_obs)
                    else:
                        obs = raw_obs
                    res = self.policy.act(obs, deterministic=True)
                    action = res[0]
                    raw_obs_next, reward, done, info = self.env.step(action)
                    if render:
                        self.env.render()
                    time.sleep(sleep)

                    if done:
                        break

                    if "cost" in info:
                        ep_cost += info["cost"]

                    ep_reward += reward
                    ep_len += 1
                    total_steps += 1
                    raw_obs = raw_obs_next

                self.logger.store(EpRet=ep_reward, EpLen=ep_len, EpCost=ep_cost, tab="eval")

                # Log info about epoch
                self._log_metrics(epoch, total_steps)
            except Exception as e:
                traceback.print_exc()
                print(e)
                self.env.wait_for_terminate()

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
