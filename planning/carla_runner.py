from curses import raw
import time
from copy import deepcopy
import os.path as osp
import gym
import gym_carla

import carla

import torch
from tqdm import tqdm

from planning.safe_rl.policy import DDPG, PPO, SAC, TD3, PPOLagrangian, SACLagrangian, DDPGLagrangian, TD3Lagrangian
from planning.safe_rl.policy.bev_policy import PPO_BEV, DDPG_BEV, SAC_BEV, TD3_BEV
from planning.safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from planning.safe_rl.util.run_util import load_config, find_config_dir, find_model_path, setup_eval_configs
from planning.safe_rl.util.torch_util import export_device_env_variable, seed_torch
from planning.safe_rl.worker import OffPolicyWorker, OnPolicyWorker
from planning.env_wrapper import carla_env

from scenario_runner.srunner.scenario_manager.carla_data_provider import CarlaDataProvider

import threading

class MyThread(threading.Thread):
    def __init__(self, threadId, epochs, sleep, render, config, world, function):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.config = config
        self.function = function
        self.epochs = epochs
        self.sleep = sleep
        self.render = render
        self.world = world
    def run(self):
        self.function(config=self.config, world=self.world, epochs=self.epochs, sleep=self.sleep, render=self.render)


class CarlaRunner:
    '''
    Main entry that coodrinate learner and worker
    '''
    # First element is the policy class while the second is whether it is an on-policy algorithm
    POLICY_LIB = {
        "ppo": (PPO, True, OnPolicyWorker),
        # "ppo_lag": (PPOLagrangian, True, OnPolicyWorker),
        "sac": (SAC, False, OffPolicyWorker),
        # "sac_lag": (SACLagrangian, False, OffPolicyWorker),
        "td3": (TD3, False, OffPolicyWorker),
        # "td3_lag": (TD3Lagrangian, False, OffPolicyWorker),
        "ddpg": (DDPG, False, OffPolicyWorker),
        # "ddpg_lag": (DDPGLagrangian, False, OffPolicyWorker),
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
                 pretrain_dir=None,
                 load_dir=None,
                 data_dir=None,
                 verbose=True,
                 continue_from_epoch=0,
                 obs_type=0,
                 port=2000,
                 traffic_port=8000,
                 **kwarg) -> None:
        seed_torch(seed)
        torch.set_num_threads(threads)
        export_device_env_variable(device, id=device_id)

        self.map_town_config = kwarg['map_town_config']
        print(self.map_town_config)

        self.episode_rerun_num = episode_rerun_num
        self.sample_episode_num = sample_episode_num
        self.evaluate_episode_num = evaluate_episode_num
        self.pretrain_dir = pretrain_dir
        self.continue_from_epoch = continue_from_epoch
        self.obs_type = obs_type

        # Instantiate environment
        self.port = port
        self.traffic_port = traffic_port
        self.obs_type = obs_type
        self.env = carla_env(obs_type, port, traffic_port)
        self.env.seed(seed)

        mode = mode.lower()
        if mode == "eval":
            # Read some basic env and model info from the dir configs
            assert load_dir is not None, "The load_path parameter has not been specified!!!"
            model_path, self.continue_from_epoch, policy, timeout_steps, policy_config = setup_eval_configs(
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
        # self.epoch = self.continue_from_epoch
        self.verbose = verbose
        if mode == "train" and "cost_limit" in self.policy_config:
            self.cost_limit = self.policy_config["cost_limit"]
        else:
            self.cost_limit = 1e3

        # Fred 2022.12.3
        # init client
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.trafficManager = self.client.get_trafficmanager(traffic_port)
        self.trafficManager.set_global_distance_to_leading_vehicle(1.0)
        self.trafficManager.set_synchronous_mode(True)


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

        if self.pretrain_dir is not None and find_config_dir(self.pretrain_dir) is not None:
            model_path, self.continue_from_epoch, _, _, _ = setup_eval_configs(self.pretrain_dir)
            self.policy.load_model(model_path)
        else:
            if self.pretrain_dir is not None:
                print("Didn't find model in %s" % self.pretrain_dir)
            if logger_kwargs['output_dir'] is not None and \
                    find_config_dir(logger_kwargs['output_dir']) is not None and \
                    find_model_path(osp.join(logger_kwargs['output_dir'], "model_save")) is not None:
                print(logger_kwargs['output_dir'])
                print(find_config_dir(logger_kwargs['output_dir']))
                model_path, self.continue_from_epoch, _, _, _ = setup_eval_configs(logger_kwargs['output_dir'])
                self.policy.load_model(model_path)
                self.pretrain_dir = logger_kwargs['output_dir']
            else:
                print('Training from scratch...')

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
        for epoch in tqdm(range(self.epochs)):
            if epoch <= self.continue_from_epoch:
                continue
            if self.on_policy:
                epoch_steps = self.train_one_epoch_on_policy(epoch)
            else:
                epoch_steps = self.train_one_epoch_off_policy(epoch)
            total_steps += epoch_steps

            for _ in tqdm(range(self.evaluate_episode_num)):
                self.worker.eval()

            if hasattr(self.policy, "post_epoch_process"):
                self.policy.post_epoch_process()

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state({'env': None}, epoch)
            # Log info about epoch
            self.data_dict = self._log_metrics(epoch, total_steps,
                                               time.time() - start_time, self.verbose)


    def init_world(self, town):
        world = self.client.load_world(town)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.set_traffic_manager_port(int(self.traffic_port))
        world.set_weather(carla.WeatherParameters.ClearNoon)

        return world


    def run_eval(self, epochs=10, sleep=0.01, render=True):
        for town in self.map_town_config:
            world = self.init_world(town)
            print("###### init world completed #######")
            config_lists = self.map_town_config[town]
            thread_list = []
            i = 0
            for config in config_lists:
                thread = MyThread(i, epochs, sleep, render, config, world, self.eval)
                # self.eval(epochs=epochs, sleep=sleep, render=render, config=config, world=world)
                thread_list.append(thread)
                i += 1
                if i == 1:
                    break

            for cur_thread in thread_list:
                cur_thread.start()

            time.sleep(100)

    def eval(self, config, world, epochs=10, sleep=0.01, render=True):
        # build town and config mapping map
        # for town in self.map_town_config:
        #     total_steps = 0
        #     # load world here
        #     # self.env.init_world(town)
        #     print("###### init world completed #######")
        total_steps = 0
        env = carla_env(self.obs_type, self.port, self.traffic_port, world=world)
        env.init_world()
        # config_lists = self.map_town_config[town]
        for epoch in range(epochs):
            # every epoch, different config
            # config = config_lists[epoch]
            kwargs = {"config": config}
            raw_obs, ep_reward, ep_len, ep_cost = env.reset(**kwargs), 0, 0, 0
            if render:
                env.render()
            for i in range(self.timeout_steps):
                if self.obs_type > 1:
                    obs = self.policy.process_img(raw_obs)
                else:
                    obs = raw_obs
                res = self.policy.act(obs, deterministic=True)
                action = res[0]
                raw_obs_next, reward, done, info = env.step(action)
                if render:
                    env.render()
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

    def _log_metrics(self, epoch, total_steps, time=None, verbose=True):
        # self.logger.log_tabular('CostLimit', self.cost_limit)
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TotalEnvInteracts', total_steps)
        for key in self.logger.logger_keys:
            self.logger.log_tabular(key, with_min_and_max=True, average_only=False)
        if time is not None:
            self.logger.log_tabular('Time', time)
        # data_dict contains all the keys except Epoch and TotalEnvInteracts
        data_dict = self.logger.dump_tabular(
            x_axis="TotalEnvInteracts",
            verbose=verbose,
        )
        return data_dict
