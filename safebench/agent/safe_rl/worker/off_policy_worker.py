import gym
import numpy as np
import torch
import joblib
import os
import traceback
from tqdm import tqdm
from cpprb import ReplayBuffer
from safebench.agent.safe_rl.policy.base_policy import Policy
from safebench.agent.safe_rl.util.logger import EpochLogger
from safebench.util.torch_util import to_tensor


class OffPolicyWorker:
    r'''
    Collect data based on the policy and env, and store the interaction data to data buffer.
    '''
    def __init__(self, config, logger):
        self.env = None
        self.policy = None
        self.logger = logger
        self.timeout_steps = config['timeout_steps']
        self.warmup_steps = config['warmup_steps']

        self.episode_rerun_num = config['episode_rerun_num']
        self.sample_episode_num = config['sample_episode_num']
        self.evaluate_episode_num = config['evaluate_episode_num']
        self.batch_size = config['batch_size']
        self.verbose = config['verbose']

        obs_dim = config['ego_state_dim']
        act_dim = config['ego_action_dim']
        self.obs_type = config['obs_type']

        env_dict = {
            'act': {
                'dtype': np.float32,
                'shape': act_dim
            },
            'done': {
                'dtype': np.float32,
            },
            'obs': {
                'dtype': np.float32,
                'shape': obs_dim
            },
            'obs2': {
                'dtype': np.float32,
                'shape': obs_dim
            },
            'rew': {
                'dtype': np.float32,
            }
        }
        # if "Safe" in env.spec.id:
        #     self.SAFE_RL_ENV = True
        #     env_dict["cost"] = {'dtype': np.float32}
        # else:
        #     self.SAFE_RL_ENV = False
        self.cpp_buffer = ReplayBuffer(config['buffer_size'], env_dict)

    def set_environment(self, env, agent):
        self.env = env
        self.policy = agent

        ######### Warmup phase to collect data with random policy #########
        steps = 0
        while steps < self.warmup_steps:
            steps += self.work(warmup=True)

        ######### Train the policy with warmup samples #########
        for i in range(self.warmup_steps // 2):
            self.policy.learn_on_batch(self.get_sample())

    def train_one_epoch(self, epoch):
        epoch_steps = 0
        range_instance = tqdm(
            range(self.sample_episode_num),
            desc='Collecting trajectories') if self.verbose else range(
                self.sample_episode_num)
        for i in range_instance:
            steps = self.work()
            epoch_steps += steps

        train_steps = self.episode_rerun_num * epoch_steps // self.batch_size
        range_instance = tqdm(
            range(train_steps), desc='training {}/{}'.format(
                epoch + 1, self.epochs)) if self.verbose else range(train_steps)
        for i in range_instance:
            data = self.get_sample()
            self.policy.learn_on_batch(data)

        return epoch_steps

    def work(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        raw_obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        if self.obs_type > 1:
            obs = self.policy.process_img(raw_obs)
        else:
            obs = raw_obs
        for i in range(self.timeout_steps):
            if warmup:
                action = self.env.sample_action_space()
            else:
                action, _ = self.policy.act(obs,
                                            deterministic=False,
                                            with_logprob=False)
            raw_obs_next, reward, done, info = self.env.step(action)  # assume action in [-1, 1]

            # TODO: hard code multiple results for now
            action = action[0]
            # raw_obs_next = raw_obs_next[0]
            # reward = reward[0]
            done = done[0]
            info = info[0]

            if self.obs_type > 1:
                obs_next = self.policy.process_img(raw_obs_next)
            else:
                obs_next = raw_obs_next
            self.logger.store(Act1=action[0], Act2=action[1], tab="worker")
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # done = False if ep_len == self.timeout_steps - 1 or "TimeLimit.truncated" in info else done
            # done = True if "goal_met" in info and info["goal_met"] else done
            if "cost" in info:
                cost = info["cost"]
                ep_cost += cost
                self.cpp_buffer.add(obs=obs,
                                    act=np.squeeze(action),
                                    rew=reward,
                                    obs2=obs_next,
                                    done=done,
                                    cost=cost)
            else:
                self.cpp_buffer.add(obs=obs,
                                    act=np.squeeze(action),
                                    rew=reward,
                                    obs2=obs_next,
                                    done=done)
                # raise Exception('no cost in info')
            ep_reward += reward
            ep_len += 1
            obs = obs_next
            if done:
                break
        self.logger.store(EpRet=ep_reward, EpCost=ep_cost, tab="worker")
        return ep_len

    def eval(self):
        '''
        Interact with the environment to collect data
        '''
        raw_obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        if self.obs_type > 1:
            obs = self.policy.process_img(raw_obs)
        else:
            obs = raw_obs
        for i in range(self.timeout_steps):
            action, _ = self.policy.act(obs, deterministic=True, with_logprob=False)
            raw_obs_next, reward, done, info = self.env.step(action)
            if self.obs_type > 1:
                obs_next = self.policy.process_img(raw_obs_next)
            else:
                obs_next = raw_obs_next
            if "cost" in info:
                cost = info["cost"]
                ep_cost += cost
            ep_reward += reward
            ep_len += 1
            obs = obs_next
            if done:
                break
        self.logger.store(TestEpRet=ep_reward,
                          TestEpLen=ep_len,
                          TestEpCost=ep_cost,
                          tab="eval")

    def get_sample(self):
        data = to_tensor(self.cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data

    def clear_buffer(self):
        self.cpp_buffer.clear()
