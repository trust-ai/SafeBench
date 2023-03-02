'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 16:38:00
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math
import carla
import json
import torch
import numpy as np

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_helper import get_location_in_distance_from_wp
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import *

from safebench.scenario.scenario_policy.maddpg.agent import Agent


class DynamicObjectCrossing(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        # self._ego_vehicle_distance_driven = 40
        # other vehicle parameters
        
        self._other_actor_target_velocity = 10
        # self._other_actor_max_brake = 1.0
        # self._time_to_reach = 10
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        # self._walker_yaw = 0
        self._num_lane_changes = 1
        # Note: transforms for walker and blocker
        self.transform = None
        self.transform2 = None
        
        self._actor_distance = 30
        self._sampling_radius = 2
        self._min_distance = 1.8
    
        self.STEERING_MAX=0.3
        
        self.adv_agent = Agent(chkpt_dir = 'checkpoints/standard_scenario1')
        self.adv_agent.load_models()
        self.adv_agent.eval()

        self.out_lane_thres = 2
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 20
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(DynamicObjectCrossing, self).__init__("DynamicObjectCrossing",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.trigger_distance_threshold = 20
#         self.actor_type_list.append('walker.*')
        self.actor_type_list.append('vehicle.diamondback.century')
        self.actor_type_list.append('static.prop.vendingmachine')
        self.ego_max_driven_distance = 150

        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self._other_actor_target_velocity * 2

    def initialize_route_planner(self):
        carla_map = self._wmap
        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[0].location + forward_vector),
                                           self.other_actor_transform[0].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[0].location,
                           carla.Location(self.other_actor_transform[0].location + forward_vector)]
        route = interpolate_trajectory(self.world, other_locations)
        init_waypoints = []
        for wp in route:
            init_waypoints.append(carla_map.get_waypoint(wp[0].location))
        
        print('init_waypoints')
        # for i in init_waypoints:
        #     print(i)
        self.routeplanner = RoutePlanner(self.other_actors[0], self.max_waypt, init_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        
    def _calculate_base_transform(self, _start_distance, waypoint):

        lane_width = waypoint.lane_width

        # Patches false junctions
        if self._reference_waypoint.is_junction:
            stop_at_junction = False
        else:
            stop_at_junction = True

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, stop_at_junction)
        waypoint = self._wmap.get_waypoint(location)
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_blocker(self, transform, orientation_yaw):
        """
        Spawn the blocker prop that blocks the vision from the egovehicle of the jaywalker
        :return:
        """
        # static object transform
        shift = 0.9
        x_ego = self._reference_waypoint.transform.location.x
        y_ego = self._reference_waypoint.transform.location.y
        x_cycle = transform.location.x
        y_cycle = transform.location.y
        x_static = x_ego + shift * (x_cycle - x_ego)
        y_static = y_ego + shift * (y_cycle - y_ego)
        spawn_point_wp = self.ego_vehicles[0].get_world().get_map().get_waypoint(transform.location)

        #Note: if need to change tranform for blocker, here
        self.transform2 = carla.Transform(carla.Location(x_static, y_static,
                                                         spawn_point_wp.transform.location.z + 0.3),
                                          carla.Rotation(yaw=orientation_yaw + 180))

    def initialize_actors(self):
        """
        Set a blocker that blocks ego's view on the walker
        Request a walker walk through the street when ego come
        """
        # cyclist transform
        _start_distance = 45
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint

        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                if wp_next.lane_width > 2:
                    _start_distance += 1.5
                    waypoint = wp_next
                break
            else:
                _start_distance += 1.5
                waypoint = wp_next

        while True:  # We keep trying to spawn avoiding props

            try:
                # Note: if need to change transform for walker, here
                self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)

                self._spawn_blocker(self.transform, orientation_yaw)

                break
            except RuntimeError as r:
                # We keep retrying until we spawn
                print("Base transform is blocking objects ", self.transform)
                _start_distance += 0.4
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z),
            self.transform.rotation)

        prop_disp_transform = carla.Transform(
            carla.Location(self.transform2.location.x,
                           self.transform2.location.y,
                           self.transform2.location.z),
            self.transform2.rotation)

        self.other_actor_transform.append(disp_transform)
        self.other_actor_transform.append(prop_disp_transform)

        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)

        self.reference_actor = self.other_actors[0]

        self.initialize_route_planner()
        self.initialize_waypoints()
        
    def initialize_waypoints(self):
        actor = self.other_actors[0].get_transform()
        initialized_waypoint = [actor.location.x, actor.location.y, actor.rotation.yaw]
        x = np.linspace(initialized_waypoint[0], self.target_waypoint[0], 16)
        y = np.linspace(initialized_waypoint[1], self.target_waypoint[1], 16)
        z = np.linspace(initialized_waypoint[2], self.target_waypoint[2], 16)
        self.initialized_waypoints = []
        for i in range(len(x)):
            self.initialized_waypoints.append([x[i],y[i],z[i]])
            
    def overwrite_waypoints(self):
        location = self.other_actors[0].get_transform().location
        location = np.array([location.x, location.y])
        max_index = -1
        for i, waypoint in enumerate(self.initialized_waypoints):
            if np.linalg.norm(location-np.array(waypoint[:2])) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index - 1):
                self.initialized_waypoints.pop(0)
        self.waypoints = self.initialized_waypoints
        
    def update_behavior(self):
        """
        the walker starts crossing the road
        """
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        self.overwrite_waypoints()

        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
            action = self.adv_agent.actor.forward(new_state).detach().cpu().numpy()[0].clip(-1,1)
            ## train ##
#             action = self.adv_agent.choose_action(state)
            print("action", action)
            current_velocity = (action[0]+1)/2 * self._other_actor_max_velocity
        self.scenario_operation.go_straight(current_velocity, 0, steering = action[1] * self.STEERING_MAX)
        self.step += 1
                                                
    def _get_obs(self):
        self.ego = self.other_actors[0]
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
            lateral_dis, -delta_yaw, speed, int(self.vehicle_front),
            (ego_x, ego_y), ego_yaw, acceleration
        ],
                         dtype=object)
        
        ### Relative Postion ###
        ego_location = self.ego_vehicles[0].get_transform().location
        ego_location = np.array([ego_location.x, ego_location.y])

        actor = self.other_actors[0]
        sv_location = actor.get_transform().location
        sv_location = np.array([sv_location.x, sv_location.y])

        rel_ego_location = ego_location - sv_location
        sv_forward_vector = actor.get_transform().rotation.get_forward_vector()
        sv_forward_vector = np.array([sv_forward_vector.x, sv_forward_vector.y])

        projection_x = sum(rel_ego_location * sv_forward_vector)
        projection_y = np.linalg.norm(rel_ego_location - projection_x * sv_forward_vector) * \
                                np.sign(np.cross(sv_forward_vector, rel_ego_location))

        ego_forward_vector = self.ego_vehicles[0].get_transform().rotation.get_forward_vector()
        ego_forward_vector = np.array([ego_forward_vector.x, ego_forward_vector.y])
        rel_ego_yaw = np.arcsin(np.cross(sv_forward_vector, ego_forward_vector))
        
        state = np.concatenate((state[:4], \
                                 np.array([rel_ego_yaw, projection_x, projection_y]))).astype(float)
        return state
                                                
    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.other_actors[0].get_velocity()

        # reward for steering:
        r_steer = -self.other_actors[0].get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.other_actors[0])
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self._other_actor_max_velocity:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = -abs(self.other_actors[0].get_control().steer) * lspeed_lon**2
        r = 0.1
        r = 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat + 0.1
        
        return r
        
    def get_state(self, actor, waypoints, idx = 2):
        r''' 
        Return the state for a specific actor.
        '''
        ego_trans = actor.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(waypoints, ego_x, ego_y, idx)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = actor.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        ### For Prediction, we also need (ego_x, ego_y), ego_yaw, acceleration ###
        state = np.array([
            lateral_dis, -delta_yaw, speed,
        ],
                         dtype=object) 
        return state

    def actor_forward(self, actor, obs, deterministic=False, with_logprob=True):
        r''' 
        Return action distribution and action log prob [optional] for SAC.
        @param obs, (tensor), [batch, obs_dim]
        @return a, (tensor), [batch, act_dim]
        @return logp, (tensor or None), (batch,)
        '''
        a, logp = actor(obs, deterministic, with_logprob)
        return a * self.act_lim, logp
    
    def check_stop_condition(self):
        """
        Now use distance actor[0] runs
        """
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.25 * lane_width * self._num_lane_changes)
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.transform)
        if cur_distance > 0.6 * lane_width:
            return True
        return False

    def _create_behavior(self):
        pass