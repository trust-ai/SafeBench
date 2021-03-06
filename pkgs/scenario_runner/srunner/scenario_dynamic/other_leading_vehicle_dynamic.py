#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Other Leading Vehicle scenario:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to decelerate.
The ego vehicle has to react accordingly by changing lane to avoid a
collision and follow the leading car in other lane. The scenario ends
either via a timeout, or if the ego vehicle drives some distance.
"""

import carla

from srunner.AdditionTools.scenario_operation import ScenarioOperation
from srunner.AdditionTools.scenario_utils import calculate_distance_transforms
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.scenario_helper import get_waypoint_in_distance
from srunner.scenario_dynamic.basic_scenario_dynamic import BasicScenarioDynamic


class OtherLeadingVehicleDynamic(BasicScenarioDynamic):

    """
    This class holds everything required for a simple "Other Leading Vehicle"
    scenario involving a user controlled vehicle and two other actors.
    Traffic Scenario 05

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 35
        self._second_vehicle_location = self._first_vehicle_location + 1
        self._ego_vehicle_drive_distance = self._first_vehicle_location * 4
        self._first_vehicle_speed = 12
        self._second_vehicle_speed = 12
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 will be safe

        self.need_decelerate = False

        super(OtherLeadingVehicleDynamic, self).__init__("VehicleDeceleratingInMultiLaneSetUpDynamic",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.actor_type_list.append('vehicle.nissan.patrol')
        self.actor_type_list.append('vehicle.audi.tt')
        self.trigger_distance_threshold = 35
        self.other_actor_speed = []
        self.other_actor_speed.append(self._first_vehicle_speed)
        self.other_actor_speed.append(self._second_vehicle_speed)
        self.ego_max_driven_distance = 200

    def initialize_actors(self):
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.other_actor_transform.append(second_vehicle_transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        # self.reference_actor = self.other_actors[1]
        self.reference_actor = self.other_actors[0]

        self._first_actor_transform = first_vehicle_transform
        # self.second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
        #                                                second_vehicle_waypoint.transform.rotation)

    def update_behavior(self):
        """
        Just make two vehicles move forward with specific speed
        At specific point, vehicle in front of ego will decelerate
        other_actors[0] is the vehicle before the ego
        """
        # cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.ego_vehicles[0]),
        #                                              CarlaDataProvider.get_transform(self.other_actors[1]))
        cur_distance = calculate_distance_transforms(self.other_actor_transform[0],
                                                     CarlaDataProvider.get_transform(self.other_actors[0]))

        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                # print("start to decelerate")
                # print("cur actor speed: ", CarlaDataProvider.get_velocity(self.other_actors[i]))
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)


    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass






    # def _initialize_actors(self, config):
    #     """
    #     Custom initialization
    #     """
    #     first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
    #     second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
    #     second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
    #
    #     first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
    #                                               first_vehicle_waypoint.transform.rotation)
    #     second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
    #                                                second_vehicle_waypoint.transform.rotation)
    #
    #     first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', first_vehicle_transform)
    #     second_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.tt', second_vehicle_transform)
    #
    #     self.other_actors.append(first_vehicle)
    #     self.other_actors.append(second_vehicle)
    #
    #     self._first_actor_transform = first_vehicle_transform
    #     self._second_actor_transform = second_vehicle_transform
    #
    # def _create_behavior(self):
    #     """
    #     The scenario defined after is a "other leading vehicle" scenario. After
    #     invoking this scenario, the user controlled vehicle has to drive towards the
    #     moving other actors, then make the leading actor to decelerate when user controlled
    #     vehicle is at some close distance. Finally, the user-controlled vehicle has to change
    #     lane to avoid collision and follow other leading actor in other lane to end the scenario.
    #     If this does not happen within 90 seconds, a timeout stops the scenario or the ego vehicle
    #     drives certain distance and stops the scenario.
    #     """
    #     # start condition
    #     driving_in_same_direction = py_trees.composites.Parallel("All actors driving in same direction",
    #                                                              policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #     leading_actor_sequence_behavior = py_trees.composites.Sequence("Decelerating actor sequence behavior")
    #
    #     # both actors moving in same direction
    #     keep_velocity = py_trees.composites.Parallel("Trigger condition for deceleration",
    #                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #     keep_velocity.add_child(WaypointFollower(self.other_actors[0], self._first_vehicle_speed, avoid_collision=True))
    #     keep_velocity.add_child(InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 55))
    #
    #     # Decelerating actor sequence behavior
    #     decelerate = self._first_vehicle_speed / 3.2
    #     leading_actor_sequence_behavior.add_child(keep_velocity)
    #     leading_actor_sequence_behavior.add_child(WaypointFollower(self.other_actors[0], decelerate,
    #                                                                avoid_collision=True))
    #     # end condition
    #     ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_drive_distance)
    #
    #     # Build behavior tree
    #     sequence = py_trees.composites.Sequence("Scenario behavior")
    #     parallel_root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #
    #     parallel_root.add_child(ego_drive_distance)
    #     parallel_root.add_child(driving_in_same_direction)
    #     driving_in_same_direction.add_child(leading_actor_sequence_behavior)
    #     driving_in_same_direction.add_child(WaypointFollower(self.other_actors[1], self._second_vehicle_speed,
    #                                                          avoid_collision=True))
    #
    #     sequence.add_child(ActorTransformSetter(self.other_actors[0], self._first_actor_transform))
    #     sequence.add_child(ActorTransformSetter(self.other_actors[1], self._second_actor_transform))
    #     sequence.add_child(parallel_root)
    #     sequence.add_child(ActorDestroy(self.other_actors[0]))
    #     sequence.add_child(ActorDestroy(self.other_actors[1]))
    #
    #     return sequence
    #
    # def _create_test_criteria(self):
    #     """
    #     A list of all test criteria will be created that is later used
    #     in parallel behavior tree.
    #     """
    #     criteria = []
    #
    #     collision_criterion = CollisionTest(self.ego_vehicles[0])
    #     criteria.append(collision_criterion)
    #
    #     return criteria
    #
    # def __del__(self):
    #     self.remove_all_actors()
