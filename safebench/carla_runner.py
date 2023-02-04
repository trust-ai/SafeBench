import carla
import pygame

from safebench.agent import AGENT_LIST
from safebench.gym_carla.env_wrapper import carla_env
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider


class CarlaRunner(object):
    """ Main body to coordinate agents and scenarios. """
    def __init__(self, agent_config, scenario_config):
        self.mode = scenario_config['mode'].lower()
        self.num_scenario = scenario_config['num_scenario']
        self.num_episode = scenario_config['num_episode']
        self.map_town_config = scenario_config['map_town_config']
        self.obs_type = agent_config['obs_type'] # the observation type is determined by the ego agent
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']

        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        self.frame_skip = scenario_config['frame_skip']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None

        # for visualization
        self.display_size = 256
        self.obs_range = 32
        self.d_behind = 12

        # pass info from scenario to agent
        agent_config['action_dim'] = scenario_config['ego_action_dim']
        agent_config['state_dim'] = scenario_config['ego_state_dim']

        # prepare ego agent
        self.agent = AGENT_LIST[agent_config['agent_name']](agent_config)
        if self.mode in ['eval', 'train_scenario'] or self.continue_agent_training:
            self.agent.load_model()
        if self.mode in ['eval', 'train_scenario']:
            self.agent.set_mode('eval')
        else:
            self.agent.set_mode('train')

    def _init_world(self, town):
        print("######## initializeing carla world ########")
        # TODO: before init world, clear all things
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self, num_envs):
        print("######## initializeing pygame birdeye renderer ########")
        pygame.init()
        # TODO: hide the window if does not want to pop up it
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF # | pygame.HIDDEN
        self.display = pygame.display.set_mode((self.display_size * 3, self.display_size * num_envs), flag)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }

        # initialize the render for genrating observation and visualization
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params)

    def run(self):
        for town in self.map_town_config.keys():
            # initialize town
            self._init_world(town)
            # initialize the renderer
            self._init_renderer(self.num_scenario)
            config_lists = self.map_town_config[town]

            # create and reset scenarios
            env_list = []
            for s_i in range(self.num_scenario):
                env = carla_env(self.obs_type, birdeye_render=self.birdeye_render, display=self.display, world=self.world)
                env.create_ego_object()
                env_list.append(env)
                # load model for scenarios 
                if self.mode in ['eval', 'train_agent'] or self.continue_scenario_training:
                    env.load_model()

            for e_i in range(self.num_episode):
                # reset envs
                onestep_info_list = []
                for s_i in range(self.num_scenario):
                    config = config_lists[s_i]
                    obs = env_list[s_i].reset(config=config, env_id=s_i, num_scenario=self.num_scenario)
                    onestep_info_list.append([obs, 0, 0])
                info_list = [onestep_info_list]

                # start the loop
                finished_env = set()
                while True: 
                    if len(finished_env) == self.num_scenario:
                        print("All scenarios are completed. Prepare for exiting")
                        break
                    
                    # get action from ego agent (assume using one batch)
                    actions_list = self.agent.get_action(info_list[-1])

                    # apply action to env and get obs
                    onestep_info_list = self._run_one_step(env_list=env_list, actions_list=actions_list, finished_env=finished_env)
                    info_list.append(onestep_info_list)

                    # train or test
                    if self.mode == 'train_agent':
                        self.agent.add_buffer(info_list, actions_list)
                        self.agent.train_model()
                        self.agent.save_model()
                    elif self.mode == 'train_scenario':
                        for k in range(len(env_list)):
                            env_list[k].add_buffer(info_list, actions_list)
                            env_list[k].train_model()
                            env_list[k].save_model()

    def _run_one_step(self, env_list, actions_list, finished_env):
        for _ in range(self.frame_skip):
            # apply action
            for j in range(len(env_list)):
                env_list[j].step_before_tick(actions_list[j])

            # tick all scenarios
            self.world.tick()

        # collect new observation of one frame
        onestep_info_list = []
        for e_i in range(len(env_list)):
            env = env_list[e_i]
            obs, reward, done, info = env.step_after_tick()
            
            # check wether env is done
            if done:
                env.is_running = False
            if not env.is_running and env not in finished_env:
                finished_env.add(env)
            
            # update infomation
            cost = info['cost'] if 'cost' in info.keys() else 0
            onestep_info_list.append([obs, reward, cost])
        return onestep_info_list
