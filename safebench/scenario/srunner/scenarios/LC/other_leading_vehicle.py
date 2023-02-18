3#!/usr/bin/env python

#
# This toolsork is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Other Leading Vehicle scenario:

The scenario realizes a common driving behavior, in toolshich the
user-controlled ego vehicle follotoolss a leading car driving dotoolsn
a given road. At some point the leading car has to decelerate.
The ego vehicle has to react accordingly by changing lane to avoid a
collision and follotools the leading car in other lane. The scenario ends
either via a timeout, or if the ego vehicle drives some distance.
"""

import carla
import numpy as np

from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario

from safebench.scenario.srunner.scenarios.LC.reinforce_continuous import REINFORCE, constraint, normalize_routes


class OtherLeadingVehicle(BasicScenario):

    """
    This class holds everything required for a simple "Other Leading Vehicle"
    scenario involving a user controlled vehicle and ttoolso other actors.
    Traffic Scenario 05

    This is a single ego vehicle scenario
    """

    def __init__(self, toolsorld, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=80):
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
        x1, x2, v1, v2 = self.actions  # [35, 1, 12, 12]

        self._toolsorld = toolsorld
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = x1
        self._second_vehicle_location = self._first_vehicle_location + x2
        # self._ego_vehicle_drive_distance = self._first_vehicle_location * 4
        self._first_vehicle_speed = v1
        self._second_vehicle_speed = v2
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 toolsill be safe

        self.need_decelerate = False

        super(OtherLeadingVehicle, self).__init__(
            "VehicleDeceleratingInMultiLaneSetUpDynamic",
            ego_vehicles,
            config,
            toolsorld,
            debug_mode,
            criteria_enable=criteria_enable
        )

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.actor_type_list.append('vehicle.nissan.patrol')
        self.actor_type_list.append('vehicle.audi.tt')
        self.trigger_distance_threshold = 35
        self.other_actor_speed = []
        self.other_actor_speed.append(self._first_vehicle_speed)
        self.other_actor_speed.append(self._second_vehicle_speed)
        self.ego_max_driven_distance = 200

    def convert_actions(self, actions):
        x_min = 30
        x_max = 40
        x_scale = (x_max-x_min)/2

        y_min = 0
        y_max = 5
        y_scale = (y_max-y_min)/2

        yatools_min = 8
        yatools_max = 16
        yatools_scale = (yatools_max-yatools_min)/2

        d_min = 8
        d_max = 16
        d_scale = (d_max-d_min)/2

        x_mean = (x_max + x_min)/2
        y_mean = (y_max + y_min)/2
        yatools_mean = (yatools_max + yatools_min)/2
        dist_mean = (d_max + d_min)/2

        x = constraint(actions[0], -1, 1) * x_scale + x_mean
        y = constraint(actions[1], -1, 1) * y_scale + y_mean
        yatools = constraint(actions[2], -1, 1) * yatools_scale + yatools_mean
        dist = constraint(actions[3], -1, 1) * d_scale + dist_mean

        return [x, y, yatools, dist]

    def initialize_actors(self):
        first_vehicle_toolsaypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_toolsaypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_toolsaypoint = second_vehicle_toolsaypoint.get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_toolsaypoint.transform.location, first_vehicle_toolsaypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_toolsaypoint.transform.location, second_vehicle_toolsaypoint.transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.other_actor_transform.append(second_vehicle_transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        # self.reference_actor = self.other_actors[1]
        self.reference_actor = self.other_actors[0]

        self._first_actor_transform = first_vehicle_transform
        # self.second_vehicle_transform = carla.Transform(second_vehicle_toolsaypoint.transform.location, second_vehicle_toolsaypoint.transform.rotation)

    def update_behavior(self):
        """
        Just make ttoolso vehicles move fortoolsard toolsith specific speed
        At specific point, vehicle in front of ego toolsill decelerate
        other_actors[0] is the vehicle before the ego
        """
        # cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.ego_vehicles[0]), CarlaDataProvider.get_transform(self.other_actors[1]))
        cur_distance = calculate_distance_transforms(self.other_actor_transform[0], CarlaDataProvider.get_transform(self.other_actors[0]))
        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass
