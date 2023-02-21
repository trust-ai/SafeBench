'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-20 20:30:35
Description: 
'''

import gym
import numpy as np
import pygame
from safebench.gym_carla.buffer import ReplayBuffer


class VectorWrapper():
    """ The interface to control a list of environments"""
    def __init__(self, agent_config, scenario_config, world, birdeye_render, display, logger, scenario_type):
        self.world = world
        self.num_scenario = scenario_config['num_scenario']
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']
        self.obs_type = agent_config['obs_type']   # the observation type is determined by the ego agent
        self.render = scenario_config['render']
        self.scenario_type = scenario_type
        self.replay_buffer = None

        self.env_list = []
        self.action_space_list = []
        for i in range(self.num_scenario):
            if i == 0:
                env = carla_env(self.obs_type, birdeye_render=birdeye_render, display=display, world=world, ROOT_DIR=self.ROOT_DIR, logger=logger, scenario_type=scenario_type, first_env=True)
            else:
                env = carla_env(self.obs_type, birdeye_render=birdeye_render, display=display, world=world, ROOT_DIR=self.ROOT_DIR, logger=logger, scenario_type=scenario_type)
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

    def reset(self, scenario_configs, scenario_type=None):
        if scenario_type is None:
            scenario_type = self.scenario_type
        self.replay_buffer = ReplayBuffer(self.num_scenario)
        # create scenarios and ego vehicles
        obs_list = []
        for s_i in range(len(scenario_configs)):
            config = scenario_configs[s_i]
            obs = self.env_list[s_i].reset(config=config, env_id=s_i, scenario_type=scenario_type)
            obs_list.append(obs)
            self.replay_buffer.save_init_obs(s_i, obs)

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True


        # return obs
        return self.obs_postprocess(obs_list)

    def step(self, ego_actions, critic_value=None, log_prob=None):
        """
            ego_actions: [num_alive_scenario, ego_action_dim]
        """
        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                self.replay_buffer.save_current_action(e_i, ego_actions[action_idx])
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                self.env_list[e_i].step_before_tick(processed_action)
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

    def step_before_tick(self, ego_action):
        self._env.step_before_tick(ego_action=ego_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _build_obs_space(self):
        if self.obs_type == 0:
            # 4 state space
            obs_dim = 4
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 1:
            # 11 state space
            obs_dim = 11
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 4 state space + bev
            obs_dim = 256  # TODO: Tune here
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


params = {
    'display_size': 256,                    # screen size of bird-eye render
    'max_past_step': 1,                     # the number of past steps to draw
    'discrete': False,                      # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],       # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],     # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'max_episode_step': 500,                # maximum timesteps per episode
    'max_waypt': 12,                        # maximum number of waypoints
    'obs_range': 32,                        # observation range (meter)
    'lidar_bin': 0.125,                     # bin size of lidar sensor (meter)
    'd_behind': 12,                         # distance behind the ego vehicle (meter)
    'out_lane_thres': 4,                    # threshold for out of lane (meter)
    'desired_speed': 8,                     # desired speed (m/s)
    'display_route': True,                  # whether to render the desired route
    'pixor_size': 64,                       # size of the pixor labels
    'pixor': False,                         # whether to output PIXOR observation
    'image_sz': 1024,
}


def carla_env(obs_type, birdeye_render=None, display=None, world=None, ROOT_DIR=None, scenario_type=None, logger=None, first_env=False):
    return EnvWrapper(
        gym.make(
            'carla-v0', 
            params=params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world, 
            ROOT_DIR=ROOT_DIR, 
            scenario_type=scenario_type,
            logger=logger,
            first_env=first_env
        ), 
        obs_type=obs_type
    )
