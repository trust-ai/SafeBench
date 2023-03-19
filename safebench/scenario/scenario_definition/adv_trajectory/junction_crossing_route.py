'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:52:46
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import json

import carla

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms


class OppositeVehicleRunningRedLight(BasicScenario):
    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):
        # Timeout of scenario in seconds
        self.timeout = timeout
        self.actor_speed = 30

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight", ego_vehicles, config, world)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)

        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")

        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 40
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150

        self.running_distance = 100
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = [(control * 2 - 1) * 3 for control in parameters]
        # self._other_actor_max_velocity = self._other_actor_target_velocity * 2
        self.total_steps = len(self.control_seq)
        self.actor_transform_list = []
        self.perturbed_actor_transform_list = []


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

        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self.running_distance
        right_vector = self.other_actor_transform[0].rotation.get_right_vector()
        for i in range(self.total_steps):
            self.actor_transform_list.append(carla.Transform(
                carla.Location(self.other_actor_transform[0].location + forward_vector * i / self.total_steps),
                self.other_actor_transform[0].rotation))
        for i in range(self.total_steps):
            self.perturbed_actor_transform_list.append(carla.Transform(
                carla.Location(self.actor_transform_list[i].location + right_vector * self.control_seq[i]),
                self.other_actor_transform[0].rotation))

        # print('other_actor_transform')
        # for i in self.other_actor_transform:
        #     print(i)
        # print('perturbed_actor_transform_list')
        # for i in self.perturbed_actor_transform_list:
        #     print(i)

    def update_behavior(self):
        target_transform = self.perturbed_actor_transform_list[self.step if self.step < self.total_steps else -1]
        # print(self.step, target_transform)
        self.step += 1  # <= 60 steps
        self.scenario_operation.drive_to_target_followlane(0, target_transform, self.actor_speed)

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
        self._target_vel = 30
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionLeftTurn, self).__init__("TurnLeftAtSignalizedJunctionDynamic",
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
        self.trigger_distance_threshold = 50
        self.ego_max_driven_distance = 150

        self.running_distance = 100
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = [(control * 2 - 1) * 3 for control in parameters]
        # self._other_actor_max_velocity = self._other_actor_target_velocity * 2
        self.total_steps = len(self.control_seq)
        self.actor_transform_list = []
        self.perturbed_actor_transform_list = []

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

        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self.running_distance
        right_vector = self.other_actor_transform[0].rotation.get_right_vector()
        for i in range(self.total_steps):
            self.actor_transform_list.append(carla.Transform(
                carla.Location(self.other_actor_transform[0].location + forward_vector * i / self.total_steps),
                self.other_actor_transform[0].rotation))
        for i in range(self.total_steps):
            self.perturbed_actor_transform_list.append(carla.Transform(
                carla.Location(self.actor_transform_list[i].location + right_vector * self.control_seq[i]),
                self.other_actor_transform[0].rotation))

        # print('other_actor_transform')
        # for i in self.other_actor_transform:
        #     print(i)
        # print('perturbed_actor_transform_list')
        # for i in self.perturbed_actor_transform_list:
        #     print(i)

    def update_behavior(self):
        # print(self.step)
        target_transform = self.perturbed_actor_transform_list[self.step if self.step < self.total_steps else -1]
        self.step += 1  # <= 50 steps
        self.scenario_operation.drive_to_target_followlane(0, target_transform, self._target_vel)

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
        self._target_vel = 30
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        self._traffic_light = None
        super(SignalizedJunctionRightTurn, self).__init__("TurnRightAtSignalizedJunctionDynamic",
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
        self.trigger_distance_threshold = 45
        self.trigger = False
        self.ego_max_driven_distance = 150

        self.running_distance = 100
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = [(control * 2 - 1) * 3 for control in parameters]
        # self._other_actor_max_velocity = self._other_actor_target_velocity * 2
        self.total_steps = len(self.control_seq)
        self.actor_transform_list = []
        self.perturbed_actor_transform_list = []

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

        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self.running_distance
        right_vector = self.other_actor_transform[0].rotation.get_right_vector()
        for i in range(self.total_steps):
            self.actor_transform_list.append(carla.Transform(
                carla.Location(self.other_actor_transform[0].location + forward_vector * i / self.total_steps),
                self.other_actor_transform[0].rotation))
        for i in range(self.total_steps):
            self.perturbed_actor_transform_list.append(carla.Transform(
                carla.Location(self.actor_transform_list[i].location + right_vector * self.control_seq[i]),
                self.other_actor_transform[0].rotation))

        # print('other_actor_transform')
        # for i in self.other_actor_transform:
        #     print(i)
        # print('perturbed_actor_transform_list')
        # for i in self.perturbed_actor_transform_list:
        #     print(i)

    def update_behavior(self):
        # print(self.step)
        target_transform = self.perturbed_actor_transform_list[self.step if self.step < self.total_steps else -1]
        self.step += 1  # <= 60 steps
        self.scenario_operation.drive_to_target_followlane(0, target_transform, self._target_vel)

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

        self.actor_speed = 30

        super(NoSignalJunctionCrossingRoute, self).__init__("NoSignalJunctionCrossing",
                                                            ego_vehicles,
                                                            config,
                                                            world,
                                                            debug_mode,
                                                            criteria_enable=criteria_enable)
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 40
        self.trigger = False

        self._actor_distance = 110
        self.ego_max_driven_distance = 150

        self.running_distance = 100
        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        # print(config.parameters)
        self.control_seq = [(control * 2 - 1) * 3 for control in parameters]
        # self._other_actor_max_velocity = self._other_actor_target_velocity * 2
        self.total_steps = len(self.control_seq)
        self.actor_transform_list = []
        self.perturbed_actor_transform_list = []

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

        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self.running_distance
        right_vector = self.other_actor_transform[0].rotation.get_right_vector()
        for i in range(self.total_steps):
            self.actor_transform_list.append(carla.Transform(
                carla.Location(self.other_actor_transform[0].location + forward_vector * i / self.total_steps),
                self.other_actor_transform[0].rotation))
        for i in range(self.total_steps):
            self.perturbed_actor_transform_list.append(carla.Transform(
                carla.Location(self.actor_transform_list[i].location + right_vector * self.control_seq[i]),
                self.other_actor_transform[0].rotation))

        # print('other_actor_transform')
        # for i in self.other_actor_transform:
        #     print(i)
        # print(self.control_seq)
        # print('perturbed_actor_transform_list')
        # for i in self.perturbed_actor_transform_list:
        #     print(i)

    def update_behavior(self):
        # print(self.step)
        target_transform = self.perturbed_actor_transform_list[self.step if self.step < self.total_steps else -1]
        self.step += 1  # <= 60 steps
        self.scenario_operation.drive_to_target_followlane(0, target_transform, self.actor_speed)

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