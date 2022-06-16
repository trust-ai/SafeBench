#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import os
import cv2
import copy
import numpy as np
import pygame
import time
import shutil
from skimage.transform import resize

import rospy
from ros_compatibility import get_service_response
from carla_ros_scenario_runner_types.msg import CarlaScenarioRunnerStatus
from carla_ros_scenario_runner_types.srv import GetEgoVehicleRoute
from carla_ros_scenario_runner_types.srv import UpdateRenderMap
from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus
import carla_common.transforms as trans


import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.render import BirdeyeRender
from gym_carla.route_planner import RoutePlanner
from gym_carla.misc import *

class CarlaEnv(gym.Env):
    """ A psuedo gym-style interface for CARLA simulator."""

    def __init__(self, params):
        # parameters
        self.role_name = params['role_name']
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        if 'task_mode' in params.keys():
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
        self.render = True
        self.global_obs_step = 0
        self.record_dir_root = os.path.join(params['record_dir'], str(int(time.time())))
        os.makedirs(self.record_dir_root, exist_ok=True)

        # Destination
        if params['task_mode'] == 'roundabout':
            self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
            self.dests = None

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

        # Connect to carla server and get world object, DON'T use load_world()
        rospy.loginfo('Connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.get_world()
        rospy.loginfo('Carla server connected!')
        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(
                np.arange(self.pixor_size),
                np.arange(self.pixor_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

        if self.render:
            self._init_renderer()
            update_render_map_service = rospy.Service('/gym_node/update_render_map', UpdateRenderMap,
                                                      self._update_render)
            rospy.loginfo('Finish initializing renderer.')
        
        # Init sensors
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        self.terminate = False
        self.time_step = 0
        self._tick = 0

        self._timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
        # For video recording
        self.scenario_count = 0

    def initialize(self):
        # Make video recording dir
        self.record_dir = os.path.join(self.record_dir_root, "scenario_" + str(int(self.scenario_count)).zfill(5))
        os.makedirs(self.record_dir, exist_ok=True)
        self.record_images_dir = os.path.join(self.record_dir, "images")
        os.makedirs(self.record_images_dir, exist_ok=True)
        self.scenario_count += 1

        self.terminate = False
        self.time_step = 0
        self._tick = 0
        # Clear sensor objects
        if self.collision_sensor is not None:
            self._stop_sensor()

        # self._clear_all_actors([
        #     'sensor.other.collision', 'sensor.lidar.ray_cast',
        #     'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker',
        #     'walker.*'
        # ])

        self.collision_hist = []
        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self._set_synchronous_mode(True)

        # Disable sync mode
        self._set_synchronous_mode(False)

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

        # find the ego vehicle by searching all vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if vehicle.attributes['role_name'] == self.role_name:
                self.ego = vehicle

        # find collision sensor
        self.collision_sensor_list = self.world.get_actors().filter('sensor.other.collision')
        if len(self.collision_sensor_list) == 0:
            raise RuntimeError('No collision sensor in the simulator')
        else:
            # if len(self.collision_sensor) > 1:
            self.collision_sensor = self.collision_sensor_list[0]
            # self._stop_sensors_list(self.collision_sensor_list)

            def get_collision_hist(event):
                impulse = event.normal_impulse
                intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                self.collision_hist.append(intensity)
                if len(self.collision_hist) > self.collision_hist_l:
                    self.collision_hist.pop(0)

            self.collision_hist = []
            self.collision_sensor.listen(lambda event: get_collision_hist(event))

        # find camera sensor
        self.camera_sensor_list = self.world.get_actors().filter('sensor.camera.rgb')
        if len(self.camera_sensor_list) == 0:
            raise RuntimeError('No camera sensor in the simulator')
        else:
            self.camera_sensor = self.camera_sensor_list[0]
            # self._stop_sensors_list(self.camera_sensor_list)
            def get_camera_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_img = array
            self.camera_sensor.listen(lambda data: get_camera_img(data))

        # find lidar 
        self.lidar_height = 2.1
        self.lidar_sensor_list = self.world.get_actors().filter('sensor.lidar.ray_cast')
        if len(self.lidar_sensor_list) == 0:
            raise RuntimeError('No LiDAR sensor in the simulator')
        else:
            self.lidar_sensor = self.lidar_sensor_list[0]
            # self._stop_sensors_list(self.lidar_sensor_list)
            def get_lidar_data(data):
                self.lidar_data = data
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        map = self.world.get_map()
        # TODO: the original planner randomly set a target point and geenrate the waypoints. However, we can directly get waypoints from ROS topic
        # One easy way is get the last point as the target and use all existing things
        response = None
        rospy.wait_for_service('/carla_data_provider/get_ego_vehicle_route')
        try:
            requester = rospy.ServiceProxy('/carla_data_provider/get_ego_vehicle_route', GetEgoVehicleRoute)
            response = requester(self.ego.id)
        except rospy.ServiceException as e:
            rospy.loginfo('Run scenario service call failed: {}'.format(e))

        init_waypoints = []
        if response is not None:
            for pose in response.ego_vehicle_route.poses:
                carla_transform = trans.ros_pose_to_carla_transform(pose.pose)
                current_waypoint = map.get_waypoint(carla_transform.location)
                init_waypoints.append(current_waypoint)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt, init_waypoints)
        self.waypoints, self.target_road_option, self.current_waypoint, self.target_waypoint, _, self.vehicle_front, self.waypoint_location_list = self.routeplanner.run_step()

        if self.render:
            # Set ego information for render
            self.birdeye_render.set_hero(self.ego, self.ego.id)

        return self._get_obs()

    def step(self, action, scenario_status):
        # When action contains acceleration and steering
        if len(action) == 2:
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
        # When action contains steer, throttle and brake
        elif len(action) == 3:
            steer = np.clip(action[0], -1, 1)
            throttle = np.clip(action[1], 0, 1)
            brake = np.clip(action[2], 0, 1)
        else:
            raise RuntimeError('Wrong number of actions.')
        # Apply control
        act = carla.VehicleControl(throttle=float(throttle),
                                   steer=float(-steer),
                                   brake=float(brake))
        self.ego.apply_control(act)
        self.world.tick()
        self._tick += 1

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
        # TODO: how to get the vehicle_front if we dont want to use the planner?
        self.waypoints, self.target_road_option, self.current_waypoint, self.target_waypoint, _, self.vehicle_front, self.waypoint_location_list = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }

        # TODO: get terminating flag from ROS topic
        # done = False
        terminal = False
        # terminal = self._terminal()
        # print(terminal, scenario_status)
        if terminal or scenario_status in [CarlaScenarioStatus.STARTING, CarlaScenarioStatus.STOPPED, CarlaScenarioStatus.SHUTTINGDOWN, CarlaScenarioStatus.ERROR]:
            done = True
            if not terminal:
                rospy.loginfo('Terminate due scenario runner')
            images = sorted(os.listdir(self.record_images_dir))
            writer = cv2.VideoWriter(os.path.join(self.record_dir, 'record.mp4'), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (self.display_size * 3, self.display_size))
            for img_name in images:
                img_path = os.path.join(self.record_images_dir, img_name)
                img = cv2.imread(img_path)
                writer.write(img)
            writer.release()
            shutil.rmtree(self.record_images_dir)
        elif scenario_status in [CarlaScenarioStatus.RUNNING]:
            done = False
        else:
            raise Exception('Unknown running status')

        self.time_step += 1
        self._timestamp = self.world.get_snapshot().timestamp.elapsed_seconds

        return self._get_obs(), self._get_reward(), done, copy.deepcopy(info)

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

    def _update_render(self, req, response=None):
        rospy.loginfo("New map requested...")
        result = True
        try:
            self.birdeye_render = BirdeyeRender(self.world, self.birdeye_render.params)
        except:
            result = False
        if req.town != self.birdeye_render.town_map.name:
            result = False
        response = get_service_response(UpdateRenderMap)
        response.result = result
        if result:
            rospy.loginfo("Map updated")
        return response

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

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
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)],
                          [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
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
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()

        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        """Get the observations."""
        if self.render:
            ## Birdeye rendering
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints

            # birdeye view with roadmap and actors
            birdeye_render_types = ['roadmap', 'actors']
            if self.display_route:
                birdeye_render_types.append('waypoints')
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
            self.display.blit(birdeye_surface, (0, 0))

        ## Lidar image generation
        point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
        x = point_cloud[:, 0:1]
        y = point_cloud[:, 1:2]
        z = point_cloud[:, 2:3]
        point_cloud = np.concatenate([y, -x, z], axis=1)

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
            wayptimg = (birdeye[:, :, 0] <= 10) * (birdeye[:, :, 1] <= 10) * (birdeye[:, :, 2] >= 240)
        else:
            wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))

        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        if self.render:
            # Display lidar image
            lidar_surface = rgb_to_display_surface(lidar, self.display_size)
            self.display.blit(lidar_surface, (self.display_size, 0))

            ## Display camera image
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (self.display_size * 2, 0))

            # Display on pygame
            pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
        acc = self.ego.get_acceleration()
        acceleration = np.sqrt(acc.x**2 + acc.y**2)
        ### For Prediction, we also need (ego_x, ego_y), ego_yaw, acceleration ###
        state = np.array([
            lateral_dis, -delta_yaw, speed, self.vehicle_front,
            (ego_x, ego_y), ego_yaw, acceleration
        ],
                         dtype=object)
        # state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])

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
        location = self.ego.get_transform().location
        node_location = self.current_waypoint.transform.location
        target_location = self.target_waypoint.transform.location
        node_forward = self.current_waypoint.transform.get_forward_vector()
        target_forward = self.target_waypoint.transform.get_forward_vector()

        obs = {
            'camera': camera.astype(np.uint8),
            'lidar': lidar.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            'state': state,
            'speed': np.float32(state[2]),
            'acc': np.float32(state[6]),
            'velocity': np.array([v.x, v.y, v.z]),
            'acceleration': np.array([acc.x, acc.y, acc.z]),
            'trajectories': self.vehicle_trajectories,
            'accelerations': self.vehicle_accelerations,
            'angular_velocities': self.vehicle_angular_velocities,
            'velocities': self.vehicle_velocities,
            'command': int(self.target_road_option.value),
            'forward_vector': np.array([forward_vector.x, forward_vector.y]),
            'location': np.array([location.x, location.y, location.z]),
            'node': np.array([node_location.x, node_location.y]),
            'target': np.array([target_location.x, target_location.y]),
            'node_forward': np.array([node_forward.x, node_forward.y]),
            'target_forward': np.array([target_forward.x, target_forward.y]),
            'rotation': np.array([ego_trans.rotation.pitch, ego_trans.rotation.yaw, ego_trans.rotation.roll]),
            'waypoint_list': np.array(self.waypoint_location_list),
            'timestamp': np.float32(self._timestamp),
            'tick': int(self._tick),
        }

        if self.render:
            try:
                pygame.image.save(self.display, os.path.join(self.record_images_dir, '%05d.png' % self.global_obs_step))
                self.global_obs_step += 1
            except:
                pass
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

        # If collides
        if len(self.collision_hist) > 0:
            rospy.loginfo('Terminate due to collision')
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            rospy.loginfo('Terminate due to max steps')
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 4:
                    rospy.loginfo('Terminate due to destination')
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            rospy.loginfo('Terminate due to out of lane')
            return True

        return False

    def _stop_sensor(self):
        self.collision_sensor.stop()
        self.lidar_sensor.stop()
        self.camera_sensor.stop()

    def _stop_sensors_list(self, sensors):
        for i in range(1, len(sensors)):
            sensors[i].stop()

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
                        actor.stop()
                    actor.destroy()
