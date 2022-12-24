#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

from scenario_runner.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from scenario_runner.srunner.scenario_dynamic.route_scenario_dynamic import RouteScenarioDynamic
from scenario_runner.srunner.scenario_manager.scenario_manager_dynamic import ScenarioManagerDynamic


class CarlaEnv2(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, port, traffic_port, params, world=None):
        # parameters
        print("in carla env 2")
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.task_mode = params['task_mode']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.lidar_bin = params['lidar_bin']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']

        if 'pixor' in params.keys():
            self.pixor = params['pixor']
            self.pixor_size = params['pixor_size']
        else:
            self.pixor = False

        # Destination
        if params['task_mode'] == 'roundabout':
            self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0],
                          [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
            self.dests = None

        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'],
                             params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(
                np.array([
                    params['continuous_accel_range'][0],
                    params['continuous_steer_range'][0]
                ]),
                np.array([
                    params['continuous_accel_range'][1],
                    params['continuous_steer_range'][1]
                ]),
                dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'camera':
            spaces.Box(low=0,
                       high=255,
                       shape=(self.obs_size, self.obs_size, 3),
                       dtype=np.uint8),
            'lidar':
            spaces.Box(low=0,
                       high=255,
                       shape=(self.obs_size, self.obs_size, 3),
                       dtype=np.uint8),
            'birdeye':
            spaces.Box(low=0,
                       high=255,
                       shape=(self.obs_size, self.obs_size, 3),
                       dtype=np.uint8),
            'state':
            spaces.Box(np.array([-2, -1, -5, 0]),
                       np.array([2, 1, 30, 1]),
                       dtype=np.float32)
        }
        if self.pixor:
            observation_space_dict.update({
                'roadmap':
                spaces.Box(low=0,
                           high=255,
                           shape=(self.obs_size, self.obs_size, 3),
                           dtype=np.uint8),
                'vh_clas':
                spaces.Box(low=0,
                           high=1,
                           shape=(self.pixor_size, self.pixor_size, 1),
                           dtype=np.float32),
                'vh_regr':
                spaces.Box(low=-5,
                           high=5,
                           shape=(self.pixor_size, self.pixor_size, 6),
                           dtype=np.float32),
                'pixor_state':
                spaces.Box(np.array([-1000, -1000, -1, -1, -5]),
                           np.array([1000, 1000, 1, 1, 20]),
                           dtype=np.float32)
            })
        self.observation_space = spaces.Dict(observation_space_dict)

        # Connect to carla server and get world object
        # print('Connecting to Carla server...')
        # self.client = carla.Client('localhost', port)
        # self.client.set_timeout(10.0)
        # TODO: here load world should be done by scenario runner
        # TODO:
        # self.world = self.client.load_world(params['town'])
        self.world = world
        self.trafficManagerPort = traffic_port
        # self.trafficManager = self.client.get_trafficmanager(traffic_port)
        # self.trafficManager.set_global_distance_to_leading_vehicle(1.0)
        # TODO:
        self.SpawnActor = carla.command.SpawnActor
        self.SetAutopilot = carla.command.SetAutopilot
        self.SetVehicleLightState = carla.command.SetVehicleLightState
        self.FutureActor = carla.command.FutureActor

        # Set weather
        # self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # # Get spawn points
        # # TODO: here vehicle spawn should be done by scenario runner
        # self.vehicle_spawn_points = list(
        #     self.world.get_map().get_spawn_points())
        # self.walker_spawn_points = []
        # for i in range(self.number_of_walkers):
        #     spawn_point = carla.Transform()
        #     loc = self.world.get_random_location_from_navigation()
        #     if loc is not None:
        #         spawn_point.location = loc
        #         self.walker_spawn_points.append(spawn_point)
        # print('Finish spawning {} walkers'.format(self.number_of_walkers))

        # # Create the ego vehicle blueprint
        # self.ego_bp = self._create_vehicle_bluepprint(
        #     params['ego_vehicle_filter'], color='49,8,8')

        # # Collision sensor
        # self.collision_hist = []  # The collision history
        # self.collision_hist_l = 1  # collision history length
        # self.collision_bp = self.world.get_blueprint_library().find(
        #     'sensor.other.collision')
        #
        # # Lidar sensor
        # self.lidar_data = None
        # self.lidar_height = 2.1
        # self.lidar_trans = carla.Transform(
        #     carla.Location(x=0.0, z=self.lidar_height))
        # self.lidar_bp = self.world.get_blueprint_library().find(
        #     'sensor.lidar.ray_cast')
        # self.lidar_bp.set_attribute('channels', '32')
        # self.lidar_bp.set_attribute('range', '5000')
        #
        # # Camera sensor
        # self.camera_img = np.zeros((self.obs_size, self.obs_size, 3),
        #                            dtype=np.uint8)
        # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        # self.camera_bp = self.world.get_blueprint_library().find(
        #     'sensor.camera.rgb')
        # # Modify the attributes of the blueprint to set image resolution and field of view.
        # self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        # self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        # self.camera_bp.set_attribute('fov', '110')
        # # Set the time in seconds between sensor captures
        # self.camera_bp.set_attribute('sensor_tick', '0.02')
        #
        # # Set fixed simulation step for synchronous mode
        # self.settings = self.world.get_settings()
        # self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        # self._init_renderer()
        # print('Finish initializing renderer')

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(
                np.arange(self.pixor_size),
                np.arange(self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

        # Init sensors
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        """for scenario runner"""
        self.scenario = None
        self.scenario_manager = None

        """for exiting"""
        self.is_running = True

        """for render outside"""
        # self.birdeye_render_vehicle_polygons = []
        # self.birdeye_render_walker_polygons = []
        # self.birdeye_render_waypoints = []
        # self.birdeye_render_types_list = []
        # self.birdeye_list = []
        # self.lidar_list = []

        # self.render_result = {}
        self.render_result = []

        # self.count_step = 0



    def init_world(self):
        # self.world = self.client.load_world(town)
        # CarlaDataProvider.set_client(self.client)
        # CarlaDataProvider.set_world(self.world)
        # CarlaDataProvider.set_traffic_manager_port(int(self.trafficManagerPort))
        # self.world.set_weather(carla.WeatherParameters.ClearNoon)
        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')

        # Lidar sensor
        self.lidar_data = None
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(
            carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find(
            'sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '32')
        self.lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3),
                                   dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find(
            'sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # self._init_renderer()

        self._init_renderer()
        # print('Finish initializing renderer')

    def load_scenario(self, config):
        self.scenario = RouteScenarioDynamic(world=self.world, config=config, timeout=800)
        self.ego = self.scenario.ego_vehicles[0]
        self.scenario_manager = ScenarioManagerDynamic()
        self.scenario_manager.load_scenario(self.scenario)
        self.scenario_manager._init_scenarios()

    #TODO: get config,
    def reset(self, **kwargs):
        config = kwargs['config']
        # TODO: load scenario, spawn ego actors
        # Clear sensor objects
        if self.collision_sensor is not None:
            self._stop_sensor()
        # Delete sensors, vehicles and walkers

        self._clear_all_actors([
            'sensor.other.collision', 'sensor.lidar.ray_cast',
            'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker',
            'walker.*'
        ])

        # Disable sync mode
        #self._set_synchronous_mode(False)

        # # first load and init world
        # self.init_world(config)
        #
        # print("###### world init completed #######")

        # then load base route scenario, same time, load ego vehicle
        self.load_scenario(config)
        print("##### route scenario loading completed #####")





        # # Spawn surrounding vehicles
        # random.shuffle(self.vehicle_spawn_points)
        # count = self.number_of_vehicles
        # batch = []
        # if count > 0:
        #     for spawn_point in self.vehicle_spawn_points:
        #         blueprint = self._create_vehicle_bluepprint(
        #             'vehicle.*', number_of_wheels=[4])
        #         blueprint.set_attribute('role_name', 'autopilot')
        #         batch.append(
        #             self.SpawnActor(blueprint, spawn_point).then(
        #                 self.SetAutopilot(self.FutureActor, True,
        #                                   self.trafficManager.get_port())))
        #         count -= 1
        #         if count <= 0:
        #             break
        #     self.client.apply_batch_sync(batch, True)
        #     self.trafficManager.global_percentage_speed_difference(30.0)
        # # Spawn pedestrians
        # # TODO: here spawn actors
        # batch = []
        # walker_speed = []
        # random.shuffle(self.walker_spawn_points)
        # count = self.number_of_walkers
        # while count > 0:
        #     if self._try_spawn_random_walker_at(
        #             random.choice(self.walker_spawn_points)):
        #         count -= 1
        #
        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Get actors info list
        self.vehicle_trajectories = []
        self.vehicle_accelerations = []
        self.vehicle_angular_velocities = []
        self.vehicle_velocities = []
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        self.vehicle_velocities.append(vehicle_info_dict_list[3])

        # Spawn the ego vehicle
        # ego_spawn_times = 0
        # while True:
        #     if ego_spawn_times > self.max_ego_spawn_times:
        #         self.reset()
        #
        #     if self.task_mode == 'random':
        #         transform = random.choice(self.vehicle_spawn_points)
        #     if self.task_mode == 'roundabout':
        #         self.start = [52.1 + np.random.uniform(-5, 5), -4.2,
        #                       178.66]  # random
        #         # self.start=[52.1,-4.2, 178.66] # static
        #         transform = set_carla_transform(self.start)
        #     if self._try_spawn_ego_vehicle_at(transform):
        #         break
        #     else:
        #         ego_spawn_times += 1
        #         time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp,
                                                       carla.Transform(),
                                                       attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp,
                                                   self.lidar_trans,
                                                   attach_to=self.ego)
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp,
                                                    self.camera_trans,
                                                    attach_to=self.ego)
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

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        # self.trafficManager.set_synchronous_mode(True)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, self.target_road_option, self.current_waypoint, self.target_waypoint, _, self.vehicle_front, = self.routeplanner.run_step()

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        self.scenario_manager._running = True

        return self._get_obs()

    def step(self, action):
        # TODO: get update
        self.scenario_manager._get_update()

        self.is_running = self.scenario_manager._running


        # Calculate acceleration and steering
        if self.discrete:
            acc = self.discrete_act[0][action // self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)
        vehicleList = self.world.get_actors().filter('vehicle.*')
        # for v in vehicleList:
        #     #if v.attributes['role_name'] != "hero":
        #     self.trafficManager.auto_lane_change(v, True)
        # Apply control
        act = carla.VehicleControl(throttle=float(throttle),
                                   steer=float(-steer),
                                   brake=float(brake))
        self.ego.apply_control(act)
        #NOTE: modify for new structure
        # self.world.tick()

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

        return (self._get_obs(), self._get_reward(), self._terminal(),
                copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
    """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 -
                                self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
    """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.
    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        blueprint = self._create_vehicle_bluepprint(
            'vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        # tm_port = self.trafficManager.get_port()
        if vehicle is not None:
            # vehicle.set_autopilot(True, tm_port)
            return True
        return False

    def _try_spawn_random_walker_at(self, walker_speed, transform):
        """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        walker_bp = random.choice(
            self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find(
                'controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(
                1 + random.random()
            )  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
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
            poly_local = np.array([[l, w], [l, -w], [-l, -w],
                                   [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat(
                [[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_actor_info(self, filt):
        """Get the info of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_acceleration_dict: a dictionary containing the accelerations of specific actors.
        """
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[
                actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()

        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        """Get the observations."""

        """
        self.birdeye_render_vehicle_polygons
        self.birdeye_render_walker_polygons
        self.birdeye_render_waypoints
        self.birdeye_render_types_list
        self.birdeye_list
        self.lidar_list
        self.camera_list
        """
        # Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.walker_polygons = self.walker_polygons
        self.birdeye_render.waypoints = self.waypoints

        # self.birdeye_render_vehicle_polygons.append(self.vehicle_polygons)
        # self.birdeye_render_walker_polygons.append(self.walker_polygons)
        # self.birdeye_render_waypoints.append(self.waypoints)



        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')

        # self.birdeye_render_types_list.append(birdeye_render_types)

        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)

        # Roadmap
        if self.pixor:
            roadmap_render_types = ['roadmap']
            if self.display_route:
                roadmap_render_types.append('waypoints')
            self.birdeye_render.render(self.display, roadmap_render_types)
            roadmap = pygame.surfarray.array3d(self.display)
            roadmap = roadmap[0:self.display_size, :, :]
            roadmap = display_to_rgb(roadmap, self.obs_size)
            # Add ego vehicle
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    if abs(birdeye[i, j, 0] - 255) < 20 and abs(
                            birdeye[i, j, 1] -
                            0) < 20 and abs(birdeye[i, j, 0] - 255) < 20:
                        roadmap[i, j, :] = birdeye[i, j, :]

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)

        # self.birdeye_list.append(birdeye_surface)
        self.display.blit(birdeye_surface, (0, 0))

        ## Lidar image generation
        point_cloud = np.copy(
            np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud,
                                 (int(point_cloud.shape[0] / 4), 4))
        x = point_cloud[:, 0:1]
        y = point_cloud[:, 1:2]
        z = point_cloud[:, 2:3]
        intensity = point_cloud[:, 3:4]
        point_cloud = np.concatenate([y, -x, z], axis=1)
        '''
        point_cloud = []
        # Get point cloud data
        for location in self.lidar_data:
        print(location.raw_data)
        point_cloud.append([location.x, location.y, -location.z])
        point_cloud = np.array(point_cloud)
        '''

        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        y_bins = np.arange(-(self.obs_range - self.d_behind),
                           self.d_behind + self.lidar_bin, self.lidar_bin)
        x_bins = np.arange(-self.obs_range / 2,
                           self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
        z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # Add the waypoints to lidar image
        if self.display_route:
            wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (
                birdeye[:, :, 2] >= 240)
        else:
            wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))

        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        # Display lidar image

        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        # self.lidar_list.append(lidar_surface)
        self.display.blit(lidar_surface, (self.display_size, 0))

        ## Display camera image

        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255

        camera_surface = rgb_to_display_surface(camera, self.display_size)

        # self.camera_list.append(camera_surface)
        self.display.blit(camera_surface, (self.display_size * 2, 0))

        # self.render_result["birdeye_render_vehicle_polygons"] = self.

        # Display on pygame
        # TODO: Solve pygame display problem
        # pygame.display.flip()
        cur_render_tuple = (self.vehicle_polygons, self.walker_polygons, self.waypoints, birdeye_render_types, birdeye_surface, lidar_surface, camera_surface)
        self.render_result.append(cur_render_tuple)

        # self.count_step += 1
        # print(self.count_step)

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(
            np.cross(w, np.array(np.array([np.cos(ego_yaw),
                                           np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        acc = self.ego.get_acceleration()
        acceleration = np.sqrt(acc.x**2 + acc.y**2)
        ### For Prediction, we also need (ego_x, ego_y), ego_yaw, acceleration ###
        # state = np.array([
        #     lateral_dis, -delta_yaw, speed, self.vehicle_front,
        #     (ego_x, ego_y), ego_yaw, acceleration
        # ],
        #                  dtype=object)
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

        if self.pixor:
            ## Vehicle classification and regression maps (requires further normalization)
            vh_clas = np.zeros((self.pixor_size, self.pixor_size))
            vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

            # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
            # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
            for actor in self.world.get_actors().filter('vehicle.*'):
                x, y, yaw, l, w = get_info(actor)
                x_local, y_local, yaw_local = get_local_pose(
                    (x, y, yaw), (ego_x, ego_y, ego_yaw))
                if actor.id != self.ego.id:
                    if abs(
                            y_local
                    ) < self.obs_range / 2 + 1 and x_local < self.obs_range - self.d_behind + 1 and x_local > -self.d_behind - 1:
                        x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
                            local_info=(x_local, y_local, yaw_local, l, w),
                            d_behind=self.d_behind,
                            obs_range=self.obs_range,
                            image_size=self.pixor_size)
                        cos_t = np.cos(yaw_pixel)
                        sin_t = np.sin(yaw_pixel)
                        logw = np.log(w_pixel)
                        logl = np.log(l_pixel)
                        pixels = get_pixels_inside_vehicle(
                            pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel,
                                        w_pixel),
                            pixel_grid=self.pixel_grid)
                        for pixel in pixels:
                            vh_clas[pixel[0], pixel[1]] = 1
                            dx = x_pixel - pixel[0]
                            dy = y_pixel - pixel[1]
                            vh_regr[pixel[0], pixel[1], :] = np.array(
                                [cos_t, sin_t, dx, dy, logw, logl])

            # Flip the image matrix so that the origin is at the left-bottom
            vh_clas = np.flip(vh_clas, axis=0)
            vh_regr = np.flip(vh_regr, axis=0)

            # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
            pixor_state = [
                ego_x, ego_y,
                np.cos(ego_yaw),
                np.sin(ego_yaw), speed
            ]

        forward_vector = self.ego.get_transform().get_forward_vector()
        node_forward = self.current_waypoint.transform.get_forward_vector()
        target_forward = self.target_waypoint.transform.get_forward_vector()

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
                'roadmap':
                roadmap.astype(np.uint8),
                'vh_clas':
                np.expand_dims(vh_clas, -1).astype(np.float32),
                'vh_regr':
                vh_regr.astype(np.float32),
                'pixor_state':
                pixor_state,
            })

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
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
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)
        terminate = False
        # If collides
        # if len(self.collision_hist) > 0:
        #     terminate = True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            terminate = True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0])**2 + (ego_y - dest[1])**2) < 4:
                    terminate = True

        # # If out of lane
        # dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        # if abs(dis) > self.out_lane_thres:
        #     terminate = True
        return terminate

    def _stop_sensor(self):
        self.collision_sensor.stop()
        self.lidar_sensor.stop()
        self.camera_sensor.stop()

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()
