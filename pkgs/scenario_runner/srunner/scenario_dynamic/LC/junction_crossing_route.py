#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
@Author: Shuai Wang
@e-mail: ws199807@outlook.com
All intersection related scenarios that are part of a route.
"""

from __future__ import print_function
import carla
import numpy as np

from srunner.AdditionTools.scenario_operation import ScenarioOperation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenario_dynamic.basic_scenario_dynamic import BasicScenarioDynamic
from srunner.AdditionTools.scenario_utils import calculate_distance_transforms

from srunner.scenario_dynamic.LC.reinforce_continuous import REINFORCE, constraint, normalize_routes


class OppositeVehicleRunningRedLightDynamic(BasicScenarioDynamic):
    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.agent = REINFORCE(config=config.parameters)
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        target_speed = 0.4

        route = []
        for point in self._ego_route:
            route.append([point[0].x, point[0].y])
        route = np.array(route)
        index = np.linspace(1, len(route) - 1, 30).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])
        route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
        route_norm = route_norm.astype('float32')

        actions = self.agent.deterministic_action(route_norm)

        self.actions = self.convert_actions(actions)
        self.x, delta_v, delta_dist = self.actions  # [0, 0, 0]

        print(actions)
        print(self.actions)

        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10 + delta_v

        super(OppositeVehicleRunningRedLightDynamic, self).__init__("OppositeVehicleRunningRedLightDynamic",
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
        self.trigger_distance_threshold = 35 + delta_dist
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = constraint(actions[0], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[1], -1, 1) * yaw_scale + yaw_mean
        dist = constraint(actions[2], -1, 1) * d_scale + dist_mean

        return [y, yaw, dist]


    def initialize_actors(self):
        """
        Custom initialization
        """
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        print(self._other_actor_transform)
        print(self.trigger_distance_threshold)
        forward_vector = self._other_actor_transform.rotation.get_forward_vector() * self.x
        self._other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0].transform, False, True)

        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")

        traffic_light_other.set_state(carla.TrafficLightState.Red)
        traffic_light_other.set_red_time(self.timeout)

    def update_behavior(self):
        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0])
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

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


class SignalizedJunctionLeftTurnDynamic(BasicScenarioDynamic):
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
        self.agent = REINFORCE(config=config.parameters)
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        target_speed = 0.4

        route = []
        for point in self._ego_route:
            route.append([point[0].x, point[0].y])
        route = np.array(route)
        index = np.linspace(1, len(route) - 1, 30).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])
        route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
        route_norm = route_norm.astype('float32')

        actions = self.agent.deterministic_action(route_norm)

        self.actions = self.convert_actions(actions)
        self.x, delta_v, delta_dist = self.actions  # [0, 0, 0]

        print(actions)
        print(self.actions)

        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12.0 + delta_v
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionLeftTurnDynamic, self).__init__("TurnLeftAtSignalizedJunctionDynamic",
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

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 45 + delta_dist
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = constraint(actions[0], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[1], -1, 1) * yaw_scale + yaw_mean
        dist = constraint(actions[2], -1, 1) * d_scale + dist_mean

        return [y, yaw, dist]

    def initialize_actors(self):
        """
        initialize actor
        """
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        forward_vector = self._other_actor_transform.rotation.get_forward_vector() * self.x
        self._other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        # self.actor_type_list.append("vehicle.diamondback.century")
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0].transform, False, True)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self):
        """
        Actor just move forward with a specific speed
        """
        for i in range(len(self.other_actors)):
            self.scenario_operation.go_straight(self._target_vel, i)

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


class SignalizedJunctionRightTurnDynamic(BasicScenarioDynamic):
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
        self.agent = REINFORCE(config=config.parameters)
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        target_speed = 0.4

        route = []
        for point in self._ego_route:
            route.append([point[0].x, point[0].y])
        route = np.array(route)
        index = np.linspace(1, len(route) - 1, 30).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])
        route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
        route_norm = route_norm.astype('float32')

        actions = self.agent.deterministic_action(route_norm)

        self.actions = self.convert_actions(actions)
        self.x, delta_v, delta_dist = self.actions  # [0, 0, 0]

        print(actions)
        print(self.actions)

        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12 + delta_v
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionRightTurnDynamic, self).__init__("TurnRightAtSignalizedJunctionDynamic",
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
        self.trigger_distance_threshold = 35 + delta_dist
        self.trigger = False
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = constraint(actions[0], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[1], -1, 1) * yaw_scale + yaw_mean
        dist = constraint(actions[2], -1, 1) * d_scale + dist_mean

        return [y, yaw, dist]

    def initialize_actors(self):
        """
        initialize actor
        """
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        forward_vector = self._other_actor_transform.rotation.get_forward_vector() * self.x
        self._other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

        traffic_light_other = CarlaDataProvider.get_next_traffic_light(config.other_actors[0].transform, False, True)
        if traffic_light_other is None:
            raise RuntimeError("No traffic light for the given location found")
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def update_behavior(self):
        """
        Actor just move forward with a specific speed
        """
        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0])
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self._target_vel, i)

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


class NoSignalJunctionCrossingRouteDynamic(BasicScenarioDynamic):
    """

    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self.agent = REINFORCE(config=config.parameters)
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        target_speed = 0.4

        route = []
        for point in self._ego_route:
            route.append([point[0].x, point[0].y])
        route = np.array(route)
        index = np.linspace(1, len(route) - 1, 30).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])
        route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
        route_norm = route_norm.astype('float32')

        actions = self.agent.deterministic_action(route_norm)

        self.actions = self.convert_actions(actions)
        self.x, delta_v, delta_dist = self.actions  # [0, 0, 0]

        print(actions)
        print(self.actions)

        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10 + delta_v

        super(NoSignalJunctionCrossingRouteDynamic, self).__init__("NoSignalJunctionCrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 35 + delta_dist
        self.trigger = False

        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def convert_actions(self, actions):
        y_scale = 5
        yaw_scale = 5
        d_scale = 5
        y_mean = yaw_mean = dist_mean = 0

        y = constraint(actions[0], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[1], -1, 1) * yaw_scale + yaw_mean
        dist = constraint(actions[2], -1, 1) * d_scale + dist_mean

        return [y, yaw, dist]

    def initialize_actors(self):
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        forward_vector = self._other_actor_transform.rotation.get_forward_vector() * self.x
        self._other_actor_transform.location += forward_vector
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]

    def update_behavior(self):
        cur_ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0])
        if cur_ego_speed and cur_ego_speed > 0.5:
            self.trigger = True
        if self.trigger:
            for i in range(len(self.other_actors)):
                self.scenario_operation.go_straight(self.actor_speed, i)

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