#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Vehicle Maneuvering In Opposite Direction:
Vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction and encroaches into another
vehicle traveling in the opposite direction.
"""
import carla

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario
from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation


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
        # parameters = [self._first_vehicle_location, self._second_vehicle_location， self._opposite_speed, self.trigger_distance_threshold]
        # parameters = [50, 30, 8, 45]
        self.parameters = config.parameters
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = self.parameters[0]
        self._second_vehicle_location = self._first_vehicle_location + self.parameters[1]
        # self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        # self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = self.parameters[2]   # m/s
        # self._source_gap = 40   # m
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        # self._source_transform = None
        # self._sink_location = None
        # self._blackboard_queue_name = 'ManeuverOppositeDirection/actor_flow_queue'
        # self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
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
        self.trigger_distance_threshold = self.parameters[3]
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

    def update_behavior(self):
        """
        first actor run in low speed
        second actor run in normal speed from oncoming route
        """
        self.scenario_operation.go_straight(self._opposite_speed, 1)



    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass
