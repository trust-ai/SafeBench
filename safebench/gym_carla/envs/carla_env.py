import copy
import numpy as np
import pygame
import random
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb, 
    rgb_to_display_surface, 
    get_lane_dis, 
    get_pos, 
    get_info,
    get_local_pose,
    get_preview_lane_dis,
    get_pixels_inside_vehicle,
    get_pixel_info,
)
from safebench.scenario.srunner.scenario_dynamic.route_scenario_dynamic import RouteScenarioDynamic
from safebench.scenario.srunner.scenario_dynamic.object_detection_dynamic import ObjectDetectionDynamic
from safebench.scenario.srunner.scenario_manager.scenario_manager_dynamic import ScenarioManagerDynamic
from safebench.scenario.srunner.tools.route_manipulation import interpolate_trajectory


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """
    def __init__(self, params, birdeye_render=None, display=None, world=None, ROOT_DIR=None):
        # parameters
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.max_episode_step = params['max_episode_step']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.display_route = params['display_route']
        self.ROOT_DIR = ROOT_DIR

        self.acc_max = params['continuous_accel_range'][1]
        self.steering_max = params['continuous_steer_range'][1]

        if 'pixor' in params.keys():
            self.pixor = params['pixor']
            self.pixor_size = params['pixor_size']
        else:
            self.pixor = False

        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            # TODO: we assume the output of NN is -1 to 1
            self.action_space = spaces.Box(
                np.array([params['continuous_accel_range'][0], params['continuous_steer_range'][0]]),
                np.array([params['continuous_accel_range'][1], params['continuous_steer_range'][1]]),
                dtype=np.float32
            )  # acc, steer
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }
        if self.pixor:
            observation_space_dict.update({
                'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
                'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
                'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
            })
        self.observation_space = spaces.Dict(observation_space_dict)

        assert world is not None, "the world passed into CarlaEnv is None"
        self.world = world
        self.birdeye_render = birdeye_render
        self.display = display
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.SetVehicleLightState = carla.command.SetVehicleLightState
        self.FutureActor = carla.command.FutureActor

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

        # Init sensors
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        """for scenario runner"""
        self.scenario = None
        self.scenario_manager = None

    def create_ego_object(self):
        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '3000')

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8) # TODO: Haohong
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    def load_scenario(self, config, env_id, scenario_type):
        # create scenario accoridng to different types
        if scenario_type in ['od']:
            self.scenario = ObjectDetectionDynamic(world=self.world, config=config, ROOT_DIR=self.ROOT_DIR, ego_id=env_id) 
        elif scenario_type in ['dev', 'standard', 'benign']:
            self.scenario = RouteScenarioDynamic(world=self.world, config=config, ego_id=env_id)
        else:
            raise NotImplementedError('{} type of scenario is not implemented.'.format(scenario_type))

        # init scenario and manager
        self.ego = self.scenario.ego_vehicles[0]
        self.scenario_manager = ScenarioManagerDynamic()
        self.scenario_manager.load_scenario(self.scenario)
        self.scenario_manager._init_scenarios()

    def reset(self, config, env_id, scenario_type):
        self.clear_up()
        print("######## loading scenario ########")
        self.load_scenario(config, env_id, scenario_type)
        self.env_id = env_id

        # change view point
        location = carla.Location(x=100, y=100, z=300)
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(location, carla.Rotation(yaw=270.0, pitch=-90.0)))

        # Get actors polygon list (for visualization)
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # Get actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # interp waypoints as init waypoints
        m = self.world.get_map()
        origin_waypoints_loc = []
        for loc in config.trajectory:
            origin_waypoints_loc.append(loc)

        _, route = interpolate_trajectory(self.world, origin_waypoints_loc, 5.0)

        # TODO: efficiency can be improved since we transform waypoints to location, and back to waypoints
        init_waypoints = []
        for node in route:
            loc = node[0].location
            init_waypoints.append(m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving))

        # TODO: check the target point of this planner
        self.routeplanner = RoutePlanner(self.ego, self.max_waypt, init_waypoints)
        self.waypoints, self.target_road_option, self.current_waypoint, self.target_waypoint, _, self.vehicle_front, = self.routeplanner.run_step()

        # TODO: applying setting can tick the world and get data from sensros
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        # self.world.tick()
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        self.scenario_manager._running = True
        return self._get_obs()

    def load_model(self):
        # TODO: load scenario policy model
        pass

    def step_before_tick(self, ego_action):
        # TODO: input an action into the scenario
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
        self.scenario_manager.get_update(timestamp)
        if isinstance(ego_action, dict):
            world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
            fov = self.camera_bp.get_attribute('fov').as_float()
            image_w, image_h = 3000., 3000.
            self.scenario_manager.evaluate(ego_action, world_2_camera, image_w, image_h, fov)
            ego_action = ego_action['ego_action']

        self.is_running = self.scenario_manager._running
        
        # Calculate acceleration and steering
        if self.discrete:
            acc = self.discrete_act[0][ego_action // self.n_steer]
            steer = self.discrete_act[1][ego_action % self.n_steer]
        else:
            acc = ego_action[0]
            steer = ego_action[1]

        # normalize and clip the action
        acc = acc * self.acc_max
        steer = steer * self.steering_max
        acc = max(min(self.acc_max, acc), -self.acc_max)
        steer = max(min(self.steering_max, steer), -self.steering_max)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)

    def step_after_tick(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # Append actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        while len(self.vehicle_trajectories) > self.max_past_step:
            self.vehicle_trajectories.pop(0)
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        while len(self.vehicle_accelerations) > self.max_past_step:
            self.vehicle_accelerations.pop(0)
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        while len(self.vehicle_angular_velocities) > self.max_past_step:
            self.vehicle_angular_velocities.pop(0)
        self.vehicle_velocities.append(vehicle_info_dict_list[3])
        while len(self.vehicle_velocities) > self.max_past_step:
            self.vehicle_velocities.pop(0)

        # route planner
        self.waypoints, self.target_road_option, self.current_waypoint, self.target_waypoint, _, self.vehicle_front, = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_vehicle_bluepprint(self,actor_filter, color=None, number_of_wheels=[4]):
        """ Create the blueprint for a specific actor type.

            Args:
                actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

            Returns:
                bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_actor_info(self, filt):
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()

        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        """ Get the observations. """
        # set ego information for birdeye_render
        self.birdeye_render.set_hero(self.ego, self.ego.id)
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # render birdeye image with the birdeye_render
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        birdeye_surface = self.birdeye_render.render(birdeye_render_types)
        birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
        center = (int(birdeye_surface.shape[0]/2), int(birdeye_surface.shape[1]/2))
        width = height = int(self.display_size/2)
        birdeye = birdeye_surface[center[0]-width:center[0]+width, center[1]-height:center[1]+height]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
            if self.display_route:
                roadmap_render_types.append('waypoints')
            self.birdeye_render.render(roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0:self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if abs(birdeye[i, j, 0] - 255) < 20 and abs(birdeye[i, j, 1] - 0) < 20 and abs(birdeye[i, j, 0] - 255) < 20:
                        roadmap[i, j, :] = birdeye[i, j, :]

        # get Lidar image
        point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
        x = point_cloud[:, 0:1]
        y = point_cloud[:, 1:2]
        z = point_cloud[:, 2:3]
        intensity = point_cloud[:, 3:4]
        point_cloud = np.concatenate([y, -x, z], axis=1)
        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
        y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
        x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
        z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # Add the waypoints to lidar image (according to the color of routes)
        # if self.display_route:
        #     wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
        # else:
        #     wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))
        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        # display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))
        # display lidar image
        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        self.display.blit(lidar_surface, (self.display_size, self.env_id*self.display_size))
        # display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size*2, self.env_id*self.display_size))

        # show image on window
        #pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        delta_yaw = np.arcsin(np.cross(w, yaw))

        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        acc = self.ego.get_acceleration()
        acceleration = np.sqrt(acc.x**2 + acc.y**2)
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

        if self.pixor:
            ## Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

            # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
            # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
            for actor in self.world.get_actors().filter('vehicle.*'):
                x, y, yaw, l, w = get_info(actor)
                x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
                if actor.id != self.ego.id:
                    if abs(y_local) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:
                        x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                            local_info=(x_local, y_local, yaw_local, l, w),
                            d_behind=self.d_behind,
                            obs_range=self.obs_range,
                            image_size=self.pixor_size
                        )
                        cos_t = np.cos(yaw_pixel)
                        sin_t = np.sin(yaw_pixel)
                        logw = np.log(w_pixel)
                        logl = np.log(l_pixel)
                        pixels = get_pixels_inside_vehicle(pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel), pixel_grid=self.pixel_grid)
                        for pixel in pixels:
                            vh_clas[pixel[0], pixel[1]] = 1
                            dx = x_pixel - pixel[0]
                            dy = y_pixel - pixel[1]
                            vh_regr[pixel[0], pixel[1], :] = np.array([cos_t, sin_t, dx, dy, logw, logl])

            # Flip the image matrix so that the origin is at the left-bottom
            vh_clas = np.flip(vh_clas, axis=0)
            vh_regr = np.flip(vh_regr, axis=0)

            # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
            pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

        #forward_vector = self.ego.get_transform().get_forward_vector()
        #node_forward = self.current_waypoint.transform.get_forward_vector()
        #target_forward = self.target_waypoint.transform.get_forward_vector()

        obs = {
            'camera': camera.astype(np.uint8),
            'lidar': lidar.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            'state': state.astype(np.float32),
            # 'trajectories': self.vehicle_trajectories,
            # 'accelerations': self.vehicle_accelerations,
            # 'angular_velocities': self.vehicle_angular_velocities,
            # 'velocities': self.vehicle_velocities,
            # 'command': int(self.target_road_option.value),
            # 'forward_vector': np.array([forward_vector.x, forward_vector.y]),
            # 'node_forward': np.array([node_forward.x, node_forward.y]),
            # 'target_forward': np.array([target_forward.x, target_forward.y]),
        }

        if self.pixor:
            obs.update({
                'roadmap': roadmap.astype(np.uint8),
                'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr': vh_regr.astype(np.float32),
                'pixor_state': pixor_state,
            })

        return obs

    def _get_reward(self):
        """ Calculate the step reward. """
        # reward for speed tracking
        v = self.ego.get_velocity()

        # TODO: reward for collision, there should be a signal from scenario
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = -abs(self.ego.get_control().steer) * lspeed_lon**2
        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat + 0.1

        return r

    def _get_cost(self):
        # cost for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
        return r_collision

    def _terminal(self):
        """ Calculate whether to terminate the current episode. """
        # terminate = False
        #
        # # If reach maximum timestep
        # if self.time_step > self.max_episode_step:
        #     terminate = True

        return not self.scenario_manager._running

    def _stop_sensor(self):
        self.collision_sensor.stop()
        self.lidar_sensor.stop()
        self.camera_sensor.stop()

    def _clear_all_actors(self, actor_filters):
        """ Clear specific actors. """
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()

    def clear_up(self):
        #Clear sensor objects
        if self.collision_sensor is not None:
            self._stop_sensor()
        
        #Delete sensors, vehicles and walkers
        self._clear_all_actors([
            'sensor.other.collision', 
            'sensor.lidar.ray_cast',
            'sensor.camera.rgb', 
            'vehicle.*',
            'controller.ai.walker',
            'walker.*'
        ])
