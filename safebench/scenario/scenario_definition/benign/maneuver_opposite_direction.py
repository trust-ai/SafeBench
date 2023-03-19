'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-04 14:22:52
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

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
        self._second_vehicle_location = self._first_vehicle_location + 30
        # self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        # self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = 8   # m/s
        # self._source_gap = 40   # m
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        # self._source_transform = None
        # self._sink_location = None
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
        self.trigger_distance_threshold = 0
        self.ego_max_driven_distance = 200


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
        self.other_actors[0].set_autopilot()
        self.other_actors[1].set_autopilot()

    def update_behavior(self):
        pass



    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass
