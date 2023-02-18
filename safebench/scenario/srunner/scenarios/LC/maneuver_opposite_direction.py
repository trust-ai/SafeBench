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
import numpy as np

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario
from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation

from safebench.scenario.srunner.scenarios.LC.reinforce_continuous import REINFORCE, constraint, normalize_routes


class ManeuverOppositeDirection(BasicScenario):
    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 06)
    This is a single ego vehicle scenario
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, obstacle_type='vehicle', timeout=120):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
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
        x1, x2, v2 = self.actions  # [50, 30, 8]

        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = x1
        self._second_vehicle_location = self._first_vehicle_location + x2
        # self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        # self._start_distance = self._first_vehicle_location * 0.9
        self._opposite_speed = v2   # m/s
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
            criteria_enable=criteria_enable
        )

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)

        self.actor_type_list.append('vehicle.nissan.micra')
        self.actor_type_list.append('vehicle.nissan.micra')
        # self.actor_type_list.append('vehicle.nissan.patrol')

        self.reference_actor = None
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200

    def convert_actions(self, actions):
        x_min = 40
        x_max = 60
        x_scale = (x_max-x_min)/2

        y_min = 20
        y_max = 40
        y_scale = (y_max-y_min)/2

        yaw_min = 6
        yaw_max = 10
        yaw_scale = (yaw_max-yaw_min)/2

        x_mean = (x_max + x_min)/2
        y_mean = (y_max + y_min)/2
        yaw_mean = (yaw_max + yaw_min)/2

        x = constraint(actions[0], -1, 1) * x_scale + x_mean
        y = constraint(actions[1], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[2], -1, 1) * yaw_scale + yaw_mean

        return [x, y, yaw]

    def initialize_actors(self):
        """
        Custom initialization
        """
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(first_actor_waypoint.transform.location, first_actor_waypoint.transform.rotation)

        self.other_actor_transform.append(first_actor_transform)
        self.other_actor_transform.append(second_actor_waypoint.transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
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
