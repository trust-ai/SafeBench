from math import cos
import gym
import numpy as np
from easydict import EasyDict as edict


CFG = edict(
    ACC_MAX=3,
    STEERING_MAX=0.3,
    OBS_TYPE=0,
    MAX_EPISODE_LEN=300,
    #NOTE: change here
    FRAME_SKIP=1,
)

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self._env = env

        self.is_running = True
        # self.spec.id = "GymCarla"
        self.spec.max_episode_steps = cfg.MAX_EPISODE_LEN
        env._max_episode_steps = cfg.MAX_EPISODE_LEN
        self._max_episode_steps = env._max_episode_steps
        self.frame_skip = cfg.FRAME_SKIP

        self._build_obs_space(cfg.OBS_TYPE)

        # build action space, assume the obs range from -1 to 1
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

        self.render_result = []

    def init_world(self):
        self._env.init_world()

    def clear_up(self):
        self._env.clear_up()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        self.ego = self._env.ego
        return self._preprocess_obs(obs)

    def step(self, action, reward, cost):
        o, r, d, info = super().step(action)
        self.is_running = self._env.is_running
        done = False
        if d:
            done = True
        r, info = self._preprocess_reward(r, info)
        o = self._preprocess_obs(o)
        reward += r
        if "cost" in info:
            cost += info["cost"]
        if "cost" in info:
            info["cost"] = cost
        self.render_result = self._env.render_result

        return o, reward, done, info, cost

    def _build_obs_space(self, obs_type):
        self.obs_type = obs_type
        if self.obs_type == 0:
            # 4 state space
            obs_dim = 4
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim, dtype=np.dtype)
        elif self.obs_type == 1:
            # 11 state space
            obs_dim = 11
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim, dtype=np.dtype)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 4 state space + bev
            obs_dim = 256  # TODO: Tune here
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim, dtype=np.dtype)
        else:
            raise NotImplementedError

    def _preprocess_obs(self, obs):
        if self.obs_type == 0:
            return obs['state'][:4].astype(np.float64)
        elif self.obs_type == 1:
            new_obs = np.array([obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
                                obs['command'], obs['forward_vector'][0], obs['forward_vector'][1],
                                obs['node_forward'][0], obs['node_forward'][1],
                                obs['target_forward'][0], obs['target_forward'][1]])
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
        # normalize and clip the action
        action = action * np.array([self.cfg.ACC_MAX, self.cfg.STEERING_MAX])
        action[0] = max(min(self.cfg.ACC_MAX, action[0]), -self.cfg.ACC_MAX)
        action[1] = max(min(self.cfg.STEERING_MAX, action[1]),
                        -self.cfg.STEERING_MAX)
        return action


params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter':
    'vehicle.lincoln*',  # filter for defining ego vehicle
    # 'port': 3030,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode':
    'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 4,  # threshold for out of lane (meter)
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
}


def carla_env(obs_type, port, traffic_port, world=None):
    CFG.OBS_TYPE = obs_type
    return EnvWrapper(gym.make('carla-v1', port=port, traffic_port=traffic_port, params=params, world=world), cfg=CFG)
