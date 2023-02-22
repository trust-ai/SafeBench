import numpy as np
import carla
import pygame

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenario_manager.scenario_trainer import ScenarioTrainer
from safebench.scenario.srunner.tools.scenario_utils import scenario_parse
from safebench.scenario.scenario_data_loader import ScenarioDataLoader

from safebench.agent import AGENT_LIST
from safebench.agent.safe_rl.agent_trainer import AgentTrainer
from safebench.util.logger import EpochLogger, setup_logger_kwargs
from safebench.util.run_util import save_video


class CarlaRunner:
    def __init__(self, agent_config, scenario_config):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.save_video = scenario_config['save_video']
        self.mode = scenario_config['mode']
        assert not self.save_video or (self.save_video and self.mode == 'eval'), "only allowed saving video in eval mode"

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.num_episode = scenario_config['num_episode']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_type = scenario_config['type_name'].split('.')[0]

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None

        self.env_params = {
            'obs_type': agent_config['obs_type'],
            'scenario_type': self.scenario_type,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'disable_lidar': True,
            'display_size': 128,                    # screen size of one bird-eye view windowd=
            'obs_range': 32,                        # observation range (meter)
            'd_behind': 12,                         # distance behind the ego vehicle (meter)
            'max_past_step': 1,                     # the number of past steps to draw
            'discrete': False,                      # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],       # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],     # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
            'max_episode_step': 500,                # maximum timesteps per episode
            'max_waypt': 12,                        # maximum number of waypoints
            'lidar_bin': 0.125,                     # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                    # threshold for out of lane (meter)
            'desired_speed': 8,                     # desired speed (m/s)
            'image_sz': 1024,                       # TODO: move to config of od scenario
        }

        # pass info from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # prepare ego agent
        if self.mode == 'eval':
            self.logger = EpochLogger(eval_mode=True)
        elif self.mode == 'train_scenario':
            self.logger = EpochLogger(eval_mode=True)
            self.trainer = ScenarioTrainer()
        elif self.mode == 'train_agent':
            logger_kwargs = setup_logger_kwargs(scenario_config['exp_name'], scenario_config['seed'], data_dir=scenario_config['data_dir'])
            self.logger = EpochLogger(**logger_kwargs)
            self.logger.save_config(agent_config)
            self.trainer = AgentTrainer(agent_config, self.logger)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")
        self.agent = AGENT_LIST[agent_config['agent_type']](agent_config, logger=self.logger)

    def _init_world(self, town):
        self.logger.log(">> Initializing carla world")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self, num_envs):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_type != 'od': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * num_envs)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * num_envs)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * num_envs)

        self.display = pygame.display.set_mode(window_size, flag)
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }

        # initialize the render for genrating observation and visualization
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)
    
    def eval(self, env, data_loader):
        num_finished_scenario = 0
        video_count = 0
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario
            # reset envs
            obss = env.reset(sampled_scenario_configs)
            rewards_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            frame_list = []
            while True:
                if env.all_scenario_done():
                    self.logger.log(">> All scenarios are completed. Prepare for exiting")
                    break

                # get action from ego agent (assume using one batch)
                ego_actions = self.agent.get_action(obss)

                # apply action to env and get obs
                obss, rewards, _, infos = env.step(ego_actions=ego_actions)

                if self.save_video:
                    one_frame = pygame.surfarray.array3d(self.display)
                    frame_list.append(one_frame.transpose(1, 0, 2))

                # accumulate reward to corresponding scenario
                reward_idx = 0
                for s_i in infos:
                    rewards_list[s_i['scenario_id']].append(rewards[reward_idx])
                    reward_idx += 1

            # clean up all things
            self.logger.log('>> Clearning up all actors')
            env.clean_up()

            # save video
            if self.save_video:
                self.logger.log('>> Saving video')
                video_name = './video/video_' + str(video_count) + '.gif'
                save_video(frame_list, video_name)
                video_count += 1

            # calculate episode reward and print
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Episode reward for batch scenario:', color='yellow')
            for s_i in rewards_list.keys():
                self.logger.log('\t Scenario ' + str(s_i) + ': ' + str(np.sum(rewards_list[s_i])), color='yellow')

    def run(self):
        # get scenario data of different maps
        maps_data = scenario_parse(self.scenario_config, self.logger)
        for town in maps_data.keys():
            # initialize town
            self._init_world(town)
            # initialize the renderer
            self._init_renderer(self.num_scenario)

            # create scenarios within the vectorized wrapper
            env = VectorWrapper(self.env_params, self.scenario_config, self.world, self.birdeye_render, self.display, self.logger)

            # prepare data loader
            data_loader = ScenarioDataLoader(maps_data[town], self.num_scenario)

            if self.mode == 'eval':
                self.eval(env, data_loader)
            elif self.mode in ['train_scenario', 'train_agent']:
                self.trainer.set_environment(env, self.agent, data_loader)
                self.trainer.train()
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def close(self):
        # check if all actors are cleaned
        actor_filters = [
            'sensor.other.collision', 
            'sensor.lidar.ray_cast',
            'sensor.camera.rgb',
            'vehicle.*',
            'walker.*',
            'controller.ai.walker',
        ]
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                self.logger.log('>> Removing agent: ' + str(actor.type_id) + '-' + str(actor.id))
                if actor.is_alive:
                    if actor.type_id.split('.')[0] in ['controller']:
                        actor.stop()
                    actor.destroy()
