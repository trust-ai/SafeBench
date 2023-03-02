'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 16:37:27
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

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import *

from safebench.scenario.scenario_policy.maddpg.agent import Agent


class OppositeVehicleRunningRedLight(BasicScenario):
    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight",
                                                             ego_vehicles,
                                                             config,
                                                             world,
                                                             debug_mode,
                                                             criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)

        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")

        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 35
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150
        
        self.STEERING_MAX=0.3
        
        self.adv_agent = Agent(chkpt_dir = './checkpoints/standard_scenario5')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self.actor_speed * 2

    def initialize_route_planner(self):
        carla_map = self.world.get_map()
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
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")

        traffic_light_other.set_state(carla.TrafficLightState.Red)
        traffic_light_other.set_red_time(self.timeout)

        self.initialize_route_planner()
    
    def update_behavior(self):
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
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

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]),
                                                     self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurn(BasicScenario):
    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario
    An actor has higher priority, ego needs to yield to
    Oncoming actor
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
            Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12.0
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionLeftTurn, self).__init__("TurnLeftAtSignalizedJunction",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0], True)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light
        self.STEERING_MAX=0.3
        self.adv_agent = Agent(chkpt_dir = './checkpoints/standard_scenario6')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 150

        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self._target_vel * 2

    def initialize_route_planner(self):
        carla_map = self.world.get_map()
        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[0].location + forward_vector),
                                           self.other_actor_transform[0].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[0].location, carla.Location(self.other_actor_transform[0].location + forward_vector)]
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
        initialize actor
        """
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        # self.actor_type_list.append("vehicle.diamondback.century")
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)
        self.initialize_route_planner()
        
    def update_behavior(self):
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
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
    
    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurn(BasicScenario):
    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario
    An actor has higher priority, ego needs to yield to
    Oncoming actor
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
            Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionRightTurn, self).__init__("TurnRightAtSignalizedJunction",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0], True)
        if self._traffic_light is None:
            raise RuntimeError("No traffic light for the given location found")
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_green_time(self.timeout)
        # other vehicle's traffic light

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 35
        self.trigger = False
        self.ego_max_driven_distance = 150
        self.STEERING_MAX=0.3
        self.adv_agent = Agent(chkpt_dir = './checkpoints/standard_scenario7')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self._target_vel * 2

    def initialize_route_planner(self):
        carla_map = self.world.get_map()
        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[0].location + forward_vector),
                                           self.other_actor_transform[0].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[0].location, carla.Location(self.other_actor_transform[0].location + forward_vector)]
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
        initialize actor
        """
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)
        self.initialize_route_planner()
        
    def update_behavior(self):
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
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
    
    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRoute(BasicScenario):
    """

    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10

        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 35
        self.trigger = False
        self.STEERING_MAX=0.3
        
        self.adv_agent = Agent(chkpt_dir = 'checkpoints/standard_scenario8')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self.actor_speed * 2

    def initialize_route_planner(self):
        carla_map = self.world.get_map()
        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[0].location + forward_vector),
                                           self.other_actor_transform[0].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[0].location, carla.Location(self.other_actor_transform[0].location + forward_vector)]
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
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        self.initialize_route_planner()
        
    def update_behavior(self):
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        with torch.no_grad():
            state = self._get_obs()
            ## eval ##
            new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
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
    
    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]),
                                                     self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False