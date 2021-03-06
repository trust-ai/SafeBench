import gym
import numpy as np
import torch
from cpprb import ReplayBuffer
from planning.safe_rl.policy.base_policy import Policy
from planning.safe_rl.util.logger import EpochLogger
from planning.safe_rl.util.torch_util import to_tensor

class OffPolicyWorker:
    r'''
    Collect data based on the policy and env, and store the interaction data to data buffer.
    '''
    def __init__(self,
                 env: gym.Env,
                 policy: Policy,
                 logger: EpochLogger,
                 batch_size=100,
                 timeout_steps=200,
                 buffer_size=1e6,
                 warmup_steps=10000,
                 **kwargs) -> None:
        self.env = env
        self.policy = policy
        self.logger = logger
        self.batch_size = batch_size
        self.timeout_steps = timeout_steps

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape

        self.obs_type = env.obs_type

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
        if "Safe" in env.spec.id:
            self.SAFE_RL_ENV = True
            env_dict["cost"] = {'dtype': np.float32}
        else:
            self.SAFE_RL_ENV = False
        self.cpp_buffer = ReplayBuffer(buffer_size, env_dict)

        ######### Warmup phase to collect data with random policy #########
        steps = 0
        while steps < warmup_steps:
            steps += self.work(warmup=True)

        ######### Train the policy with warmup samples #########
        for i in range(warmup_steps // 2):
            self.policy.learn_on_batch(self.get_sample())

    def work(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        raw_obs, ep_reward, ep_len, ep_cost = self.env.wait_for_reset(), 0, 0, 0
        if self.obs_type > 1:
            obs = self.policy.process_img(raw_obs)
        else:
            obs = raw_obs
        for i in range(self.timeout_steps):
            if warmup:
                action = self.env.action_space.sample()
            else:
                action, _ = self.policy.act(obs,
                                            deterministic=False,
                                            with_logprob=False)
            raw_obs_next, reward, done, info = self.env.step(action)
            if self.obs_type > 1:
                obs_next = self.policy.process_img(raw_obs_next)
            else:
                obs_next = raw_obs_next
            self.logger.store(Act1=action[0], Act2=action[1], tab="worker")
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if ep_len == self.timeout_steps - 1 or "TimeLimit.truncated" in info else done
            done = True if "goal_met" in info and info["goal_met"] else done
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
        raw_obs, ep_reward, ep_len, ep_cost = self.env.wait_for_reset(), 0, 0, 0
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
