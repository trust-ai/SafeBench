'''
Author: 
Email:
Date: 2023-02-16 11:20:54
LastEditTime: 2023-02-24 15:02:51
Description: 
'''

import carla

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario


class OtherLeadingVehicle(BasicScenario):
    """
        Ego vehicle follows a leading car driving down a given road. At some point the leading car has to decelerate.
        The ego vehicle has to react accordingly by changing lane to avoid a collision and follow the leading car in other lane. 
        The scenario ends either via a timeout, or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicles, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("VehicleDeceleratingInMultiLaneSetUp", ego_vehicles, config, world)

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
        self.timeout = timeout

        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 will be safe
        self.need_decelerate = False

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
        # second_vehicle_waypoint = second_vehicle_waypoint.get_right_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location, first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location, second_vehicle_waypoint.transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.other_actor_transform.append(second_vehicle_transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        # self.reference_actor = self.other_actors[1]
        self.reference_actor = self.other_actors[0]
        self._first_actor_transform = first_vehicle_transform

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        """
            Just make two vehicles move forward with specific speed
            At specific point, vehicle in front of ego will decelerate other_actors[0] is the vehicle before the ego
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        
        cur_distance = calculate_distance_transforms(self.other_actor_transform[0], CarlaDataProvider.get_transform(self.other_actors[0]))
        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.scenario_operation.go_straight(self.dece_target_speed, i)
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)

    def check_stop_condition(self):
        pass
