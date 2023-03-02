'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 16:38:02
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import torch
import numpy as np
import math
import carla
import json

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario

from safebench.scenario.tools.scenario_helper import generate_target_waypoint_in_route
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import *

from safebench.scenario.scenario_policy.maddpg.agent import Agent


def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
    Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
    Gets the driving / parking lane that is most to the right of the waypoint
    as well as the number of lane changes done
    """
    lane_changes = 0

    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next

    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
    This function filters false negative Shoulder which are in reality Parking lanes.
    These are differentiated from the others because, similar to the driving lanes,
    they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True

    return False


class VehicleTurningRoute(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn. This is the version used when the ego vehicle
    is following a given route. (Traffic Scenario 4)
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._other_actor_target_velocity = 10
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        self._num_lane_changes = 0
        
        self._actor_distance = 30
        self._sampling_radius = 2
        self._min_distance = 1.8
        
        self.STEERING_MAX=0.3
        self.adv_agent = Agent(chkpt_dir = './checkpoints/standard_scenario2')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        self.out_lane_thres = 2
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(VehicleTurningRoute, self).__init__("VehicleTurningRoute",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable,
                                                  terminate_on_failure=True)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)

        self.actor_type_list.append('vehicle.diamondback.century')

        self.reference_actor = None
        self.trigger_distance_threshold = 17
        self.ego_max_driven_distance = 180

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
        
    def initialize_actors(self):
        """
        Custom initialization
        """
        waypoint = generate_target_waypoint_in_route(self._reference_waypoint, self._ego_route)

        # Move a certain distance to the front
        start_distance = 8
        waypoint = waypoint.next(start_distance)[0]

        # Get the last driving lane to the right
        waypoint, self._num_lane_changes = get_right_driving_lane(waypoint)
        # And for synchrony purposes, move to the front a bit
        added_dist = self._num_lane_changes

        _other_actor_transform = get_opponent_transform(added_dist, waypoint, self._trigger_location)

        self.other_actor_transform.append(_other_actor_transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        
        """Also need to specify reference actor"""
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
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        self.overwrite_waypoints()

        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor([state], dtype=torch.float).to(self.adv_agent.actor.device)
            action = self.adv_agent.actor.forward(new_state).detach().cpu().numpy()[0]
            ## train ##
#             action = self.adv_agent.choose_action(state)
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
        
    def check_stop_condition(self):
        """
        This condition is just for small scenarios
        """

        return False


    def _create_behavior(self):
        pass