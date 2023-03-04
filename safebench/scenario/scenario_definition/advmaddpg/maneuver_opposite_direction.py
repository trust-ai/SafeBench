'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-04 14:21:53
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import json
import torch
import numpy as np

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import *

from safebench.scenario.scenario_policy.maddpg.agent import Agent


class ManeuverOppositeDirection(BasicScenario):

    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 06)
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, timeout=60):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 50
        self._second_vehicle_location = self._first_vehicle_location + 30
        # self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        # self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = 8   # m/s
        # self._source_gap = 40   # m
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        # self._source_transform = None
        # self._sink_location = None
        self._first_actor_transform = None
        self._second_actor_transform = None
        # self._third_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout
        # self.first_actor_speed = 0
        # self.second_actor_speed = 30
        
        self._actor_distance = self._second_vehicle_location
        self.STEERING_MAX=0.3
    
        self.adv_agent = Agent(chkpt_dir = './checkpoints/standard_scenario4')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        super(ManeuverOppositeDirection, self).__init__(
            "ManeuverOppositeDirection",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)

        self.actor_type_list.append('vehicle.nissan.micra')
        self.actor_type_list.append('vehicle.nissan.micra')
        # self.actor_type_list.append('vehicle.nissan.patrol')

        self.reference_actor = None
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200

        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self._opposite_speed * 2

    def initialize_route_planner(self):
        carla_map = self.world.get_map()
        forward_vector = self.other_actor_transform[1].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[1].location + forward_vector), self.other_actor_transform[1].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[1].location, carla.Location(self.other_actor_transform[1].location + forward_vector)]
        route = interpolate_trajectory(self.world, other_locations)
        init_waypoints = []
        for wp in route:
            init_waypoints.append(carla_map.get_waypoint(wp[0].location))
        
        print('init_waypoints')
        # for i in init_waypoints:
        #     print(i)
        self.routeplanner = RoutePlanner(self.other_actors[1], self.max_waypt, init_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()

    def initialize_actors(self):
        """
        Custom initialization
        """
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation)

        self.other_actor_transform.append(first_actor_transform)

        self.other_actor_transform.append(second_actor_waypoint.transform)

        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)

        self.reference_actor = self.other_actors[0]
        self.initialize_route_planner()

    def update_behavior(self):
        """
        first actor run in low speed
        second actor run in normal speed from oncoming route
        """
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor([state], dtype=torch.float).to(self.adv_agent.actor.device)
            action = self.adv_agent.actor.forward(new_state).detach().cpu().numpy()[0]
            ## train ##
#             action = self.adv_agent.choose_action(state)
            current_velocity = (action[0]+1)/2 * self._other_actor_max_velocity
        self.scenario_operation.go_straight(current_velocity, 1, steering = action[1] * self.STEERING_MAX)
        self.step += 1
                                                
    def _get_obs(self):
        self.ego = self.other_actors[1]
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

        actor = self.other_actors[1]
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
        self.ego = self.other_actors[1]
        v = self.ego.get_velocity()

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
        if lspeed_lon > self._other_actor_max_velocity:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = -abs(self.ego.get_control().steer) * lspeed_lon**2
        r = 0.1
        r = 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat + 0.1
        
        return r
    
    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass
