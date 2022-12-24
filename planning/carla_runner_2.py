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
from planning.env_wrapper_2 import carla_env

from gym_carla.envs.render import BirdeyeRender
from scenario_runner.srunner.scenario_manager.carla_data_provider import CarlaDataProvider

import pygame

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
        print("thread started, ", self.threadId)
        cur_ego, cur_render_result = self.function(config=self.config, world=self.world, epochs=self.epochs, sleep=self.sleep, render=self.render)
        self.ego = cur_ego
        self.render_result = cur_render_result
        print("thread done", self.threadId)


class CarlaRunner2:
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

        self.display_size = 256
        self.obs_range = 32
        self.d_behind = 12


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
        # before init world, clear all things


        world = self.client.load_world(town)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(world)
        CarlaDataProvider.set_traffic_manager_port(int(self.traffic_port))
        world.set_weather(carla.WeatherParameters.ClearNoon)

        return world

    def _init_renderer(self, num_envs):
        """Initialize the birdeye view renderer.
    """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size * num_envs),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 -
                                self.d_behind) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        # birdeye_render = BirdeyeRender(world, birdeye_params)

    def run_eval(self, epochs=10, sleep=0.01, render=True):
        for town in self.map_town_config:
            world = self.init_world(town)
            print("###### init world completed #######")
            config_lists = self.map_town_config[town]

            chosen_config = [0]

            env_list = []
            obs_list = []

            for i in chosen_config:
                # initialize env
                config = config_lists[i]
                env = carla_env(self.obs_type, self.port, self.traffic_port, world=world)
                env_list.append(env)
                env.init_world()
                kwargs = {"config": config}
                raw_obs, ep_reward, ep_len, ep_cost = env.reset(**kwargs), 0, 0, 0
                obs_list.append([raw_obs, ep_reward, ep_len, ep_cost])

                if render:
                    env.render()

            #TODO: make sure quiting for loop when all scenarios are finished
            for i in range(self.timeout_steps):
                for j in range(len(env_list)):
                    cur_env = env_list[j]

                    if not cur_env.is_running:
                        continue

                    cur_obs = obs_list[j]

                    raw_obs, ep_len, ep_reward, ep_cost, done = self.eval(cur_env, cur_obs[0], cur_obs[1], cur_obs[2], cur_obs[3])

                    if done:
                        cur_env.is_running = False
                        continue

                    obs_list[j] = [raw_obs, ep_len, ep_reward, ep_cost]
                # tick the world
                # Note: moving tick out will make ticking blocked
                world.tick()

            # display
            self._init_renderer(len(env_list))
            self.render_display(env_list, world)

    def render_display(self, env_list, world):
        print("in render display")
        # print(len(thread_list[0].render_result))
        birdeye_render_list = []

        max_len = 0

        for cur_env in env_list:
            birdeye_render = BirdeyeRender(world, self.birdeye_params)
            birdeye_render.set_hero(cur_env.ego, cur_env.ego.id)
            birdeye_render_list.append(birdeye_render)

            print(len(cur_env.render_result))

            max_len = max(len(cur_env.render_result), max_len)

        # print(self.timeout_steps)

        for i in range(max_len):
            for j in range(len(env_list)):
                if i >= len(env_list[j].render_result):
                    continue
                # if j != 0:
                #     continue
                cur_render_result = env_list[j].render_result[i]
                cur_birdeye_render = birdeye_render_list[j]

                cur_birdeye_render.vehicle_polygons = cur_render_result[0]
                cur_birdeye_render.walker_polygons = cur_render_result[1]
                cur_birdeye_render.waypoints = cur_render_result[2]

                # cur_birdeye_render.render(self.display, cur_render_result[3])

                self.display.blit(cur_render_result[4], (0, j * self.display_size))
                self.display.blit(cur_render_result[5], (self.display_size, j * self.display_size))
                self.display.blit(cur_render_result[6], (self.display_size * 2, j * self.display_size))
                pygame.display.flip()
            time.sleep(0.1)

    def eval(self, env, raw_obs, ep_reward, ep_len, ep_cost, render=True, sleep=0.01):
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

        if "cost" in info:
            ep_cost += info["cost"]

        ep_reward += reward
        ep_len += 1
        raw_obs = raw_obs_next

        return raw_obs, ep_reward, ep_len, ep_cost, done

    # def eval(self, config, world, epochs=10, sleep=0.01, render=True):
    #     # build town and config mapping map
    #     # for town in self.map_town_config:
    #     #     total_steps = 0
    #     #     # load world here
    #     #     # self.env.init_world(town)
    #     #     print("###### init world completed #######")
    #     total_steps = 0
    #     env = carla_env(self.obs_type, self.port, self.traffic_port, world=world)
    #     env.init_world()
    #     # config_lists = self.map_town_config[town]
    #     # for epoch in range(epochs):
    #     # every epoch, different config
    #     # config = config_lists[epoch]
    #     kwargs = {"config": config}
    #     raw_obs, ep_reward, ep_len, ep_cost = env.reset(**kwargs), 0, 0, 0
    #     ego = env.ego
    #     if render:
    #         env.render()
    #     for i in range(self.timeout_steps):
    #         # check if current scenario is running out
    #         if not env.is_running:
    #             return ego, env.render_result
    #         if self.obs_type > 1:
    #             obs = self.policy.process_img(raw_obs)
    #         else:
    #             obs = raw_obs
    #         res = self.policy.act(obs, deterministic=True)
    #         action = res[0]
    #         raw_obs_next, reward, done, info = env.step(action)
    #         if render:
    #             env.render()
    #         time.sleep(sleep)
    #
    #         if done:
    #             break
    #
    #         if "cost" in info:
    #             ep_cost += info["cost"]
    #
    #         ep_reward += reward
    #         ep_len += 1
    #         total_steps += 1
    #         raw_obs = raw_obs_next
    #
    #     self.logger.store(EpRet=ep_reward, EpLen=ep_len, EpCost=ep_cost, tab="eval")
    #
    #     # Log info about epoch
    #     # self._log_metrics(epoch, total_steps)
    #     return ego, env.render_result

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
