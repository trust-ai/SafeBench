'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:52:51
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla
import json

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):

    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 06)
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 obstacle_type='vehicle', timeout=120):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 50
        self._second_vehicle_location = self._first_vehicle_location + 20
        # self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        # self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = 30   # m/s
        # self._source_gap = 40   # m
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        # self._source_transform = None
        # self._sink_location = None
        # self._blackboard_queue_name = 'ManeuverOppositeDirection/actor_flow_queue'
        self._obstacle_type = obstacle_type
        self._first_actor_transform = None
        self._second_actor_transform = None
        # self._third_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        # self.first_actor_speed = 0
        # self.second_actor_speed = 30

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
        self.control_seq = [(control * 2 - 1) * 2 for control in parameters]
        self.total_steps = len(self.control_seq)
        self.actor_transform_list = []
        self.perturbed_actor_transform_list = []
        self.running_distance = 50


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

        forward_vector = self.other_actor_transform[1].rotation.get_forward_vector() * self.running_distance
        right_vector = self.other_actor_transform[1].rotation.get_right_vector()
        self.other_actor_final_transform = carla.Transform(
            self.other_actor_transform[1].location,
            self.other_actor_transform[1].rotation)
        self.other_actor_final_transform.location += forward_vector
        for i in range(self.total_steps):
            self.actor_transform_list.append(carla.Transform(
                carla.Location(self.other_actor_transform[1].location + forward_vector * i / self.total_steps),
                self.other_actor_transform[1].rotation))
        for i in range(self.total_steps):
            self.perturbed_actor_transform_list.append(carla.Transform(
                carla.Location(self.actor_transform_list[i].location + right_vector * self.control_seq[i]),
                self.other_actor_transform[1].rotation))

        # print('other_actor_transform')
        # for i in self.other_actor_transform:
        #     print(i)
        # print('perturbed_actor_transform_list')
        # for i in self.perturbed_actor_transform_list:
        #     print(i)

    def update_behavior(self):
        """
        first actor run in low speed
        second actor run in normal speed from oncoming route
        """
        # print(self.step)
        target_transform = self.perturbed_actor_transform_list[self.step if self.step < self.total_steps else -1]
        self.step += 1  # <= 50 steps
        self.scenario_operation.drive_to_target_followlane(1, target_transform, self._opposite_speed)

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass
