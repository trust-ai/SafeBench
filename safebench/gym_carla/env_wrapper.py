'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-24 00:51:01
Description: 
'''

import gym
import numpy as np
import pygame
from safebench.gym_carla.buffer import ReplayBuffer


class VectorWrapper():
    """ The interface to control a list of environments"""
    def __init__(self, env_params, scenario_config, world, birdeye_render, display, logger):
        self.world = world
        self.num_scenario = scenario_config['num_scenario']
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']  
        self.render = scenario_config['render']
        self.scenario_type = env_params['scenario_type']
        self.replay_buffer = None

        self.env_list = []
        self.action_space_list = []
        for i in range(self.num_scenario):
            env = carla_env(env_params, birdeye_render=birdeye_render, display=display, world=world, logger=logger, first_env=not i)
            self.env_list.append(env)
            self.action_space_list.append(env.action_space)

        # flags for env list 
        self.finished_env = [False] * self.num_scenario

    def load_model(self):
        for e_i in range(self.num_scenario):
            self.env_list[e_i].load_model()

    def obs_postprocess(self, obs_list):
        # assume all variables are array
        obs_list = np.array(obs_list)
        return obs_list

    def reset(self, scenario_configs, scenario_init_actions, scenario_type=None):
        if scenario_type is None:
            scenario_type = self.scenario_type
        
        # TODO: move buffer out of env
        self.replay_buffer = ReplayBuffer(self.num_scenario)

        # create scenarios and ego vehicles
        obs_list = []
        for s_i in range(len(scenario_configs)):
            config = scenario_configs[s_i]
            obs = self.env_list[s_i].reset(config=config, env_id=s_i, scenario_init_action=scenario_init_actions[s_i], scenario_type=scenario_type)
            obs_list.append(obs)
            self.replay_buffer.save_init_obs(s_i, obs)

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True

        # return obs
        return self.obs_postprocess(obs_list)

    def step(self, ego_actions, scenario_actions, critic_value=None, log_prob=None):
        """
            ego_actions: [num_alive_scenario, ego_action_dim]
            scenario_actions: [num_alive_scenario, scenario_action_dim]
        """
        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                self.replay_buffer.save_current_action(e_i, ego_actions[action_idx])
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                # TODO: pre-process scenario action
                self.env_list[e_i].step_before_tick(processed_action, scenario_actions[action_idx])
                action_idx += 1

        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()

        # collect new observation of one frame
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                obs, reward, done, info = self.env_list[e_i].step_after_tick()
                self.replay_buffer.save_step_results(
                    e_i, 
                    next_obs=obs, 
                    reward=reward, 
                    done=done, 
                    info=info,
                    critic_value=critic_value, 
                    log_prob=log_prob
                )
                info['scenario_id'] = e_i

                # check if env is done
                if done:
                    self.finished_env[e_i] = True

                # update infomation
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
        rewards = np.array(reward_list)
        dones = np.array(done_list)
        infos = np.array(info_list)

        # update pygame window
        if self.render:
            pygame.display.flip()

        return self.obs_postprocess(obs_list), rewards, dones, infos

    def sample_action_space(self):
        action = []
        for action_space in self.action_space_list:
            action.append(action_space.sample())
        return np.array(action)

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    def clean_up(self):
        # stop sensor objects
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()

        # tick to ensure that all destroy commands are executed
        self.world.tick()


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, obs_type):
        super().__init__(env)
        self._env = env

        self.is_running = False
        self.obs_type = obs_type
        self._build_obs_space()

        # build action space, assume the obs range from -1 to 1
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

    def create_ego_object(self):
        self._env.create_ego_object()

    def clear_up(self):
        self._env.clear_up()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return self._preprocess_obs(obs)

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _build_obs_space(self):
        if self.obs_type == 0:
            obs_dim = 4
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 1:
            obs_dim = 11
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 4 state space + bev
            obs_dim = 256  # TODO: should be the same as display_size
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        else:
            raise NotImplementedError

    def _preprocess_obs(self, obs):
        if self.obs_type == 0:
            return obs['state'][:4].astype(np.float64)
        elif self.obs_type == 1:
            new_obs = np.array([
                obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
                obs['command'], 
                obs['forward_vector'][0], obs['forward_vector'][1],
                obs['node_forward'][0], obs['node_forward'][1],
                obs['target_forward'][0], obs['target_forward'][1]
            ])
            return new_obs
        elif self.obs_type == 2:
            return {"img": obs['birdeye'], "states": obs['state'][:4].astype(np.float64)}
        elif self.obs_type == 3:
            return {"img": obs['camera'], "states": obs['state'][:4].astype(np.float64)}
        else:
            raise NotImplementedError

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, action):
        return action


def carla_env(env_params, birdeye_render=None, display=None, world=None, logger=None, first_env=False):
    return EnvWrapper(
        gym.make(
            'carla-v0', 
            env_params=env_params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world, 
            logger=logger,
            first_env=first_env
        ), 
        obs_type=env_params['obs_type']
    )
