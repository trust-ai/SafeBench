'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-04 21:40:47
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarios/route_scenario.py>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import traceback

import carla

from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.scenario_config import RouteScenarioConfig
from safebench.scenario.tools.route_parser import RouteParser
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.scenario.tools.scenario_utils import (
    get_valid_spawn_points, 
    convert_json_to_transform, 
    convert_json_to_actor, 
    convert_transform_to_location
)

from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
    ActorSpeedAboveThresholdTest
)

# ordinary scenario (for training)
from safebench.scenario.scenario_definition.ordinary.autopolit_background_vehicle import AutopolitBackgroundVehicle
# standard
from safebench.scenario.scenario_definition.standard.object_crash_vehicle import DynamicObjectCrossing as scenario_03_standard
from safebench.scenario.scenario_definition.standard.object_crash_intersection import VehicleTurningRoute as scenario_04_standard
from safebench.scenario.scenario_definition.standard.other_leading_vehicle import OtherLeadingVehicle as scenario_05_standard
from safebench.scenario.scenario_definition.standard.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_standard

# benign
from safebench.scenario.scenario_definition.benign.object_crash_vehicle import DynamicObjectCrossing as scenario_03_benign
from safebench.scenario.scenario_definition.benign.object_crash_intersection import VehicleTurningRoute as scenario_04_benign
from safebench.scenario.scenario_definition.benign.other_leading_vehicle import OtherLeadingVehicle as scenario_05_benign
from safebench.scenario.scenario_definition.benign.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_benign
from safebench.scenario.scenario_definition.benign.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_benign
from safebench.scenario.scenario_definition.benign.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_benign
from safebench.scenario.scenario_definition.benign.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_benign
from safebench.scenario.scenario_definition.benign.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_benign

# carla challenge
from safebench.scenario.scenario_definition.carla_challenge.object_crash_vehicle import DynamicObjectCrossing as scenario_03_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.object_crash_intersection import VehicleTurningRoute as scenario_04_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.other_leading_vehicle import OtherLeadingVehicle as scenario_05_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_carla_challenge
from safebench.scenario.scenario_definition.carla_challenge.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_carla_challenge

# LC
from safebench.scenario.scenario_definition.LC.object_crash_vehicle import DynamicObjectCrossing as scenario_03_lc
from safebench.scenario.scenario_definition.LC.object_crash_intersection import VehicleTurningRoute as scenario_04_lc
from safebench.scenario.scenario_definition.LC.other_leading_vehicle import OtherLeadingVehicle as scenario_05_lc
from safebench.scenario.scenario_definition.LC.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_lc
from safebench.scenario.scenario_definition.LC.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_lc
from safebench.scenario.scenario_definition.LC.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_lc
from safebench.scenario.scenario_definition.LC.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_lc
from safebench.scenario.scenario_definition.LC.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_lc

# AdvTraj
from safebench.scenario.scenario_definition.adv_trajectory.object_crash_vehicle import DynamicObjectCrossing as scenario_03_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.object_crash_intersection import VehicleTurningRoute as scenario_04_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.other_leading_vehicle import OtherLeadingVehicle as scenario_05_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_advtraj
from safebench.scenario.scenario_definition.adv_trajectory.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_advtraj

# AdvSim
from safebench.scenario.scenario_definition.advsim.object_crash_vehicle import DynamicObjectCrossing as scenario_03_advsim
from safebench.scenario.scenario_definition.advsim.object_crash_intersection import VehicleTurningRoute as scenario_04_advsim
from safebench.scenario.scenario_definition.advsim.other_leading_vehicle import OtherLeadingVehicle as scenario_05_advsim
from safebench.scenario.scenario_definition.advsim.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_advsim
from safebench.scenario.scenario_definition.advsim.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_advsim
from safebench.scenario.scenario_definition.advsim.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_advsim
from safebench.scenario.scenario_definition.advsim.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_advsim
from safebench.scenario.scenario_definition.advsim.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_advsim

# AdvMADDPG
from safebench.scenario.scenario_definition.advmaddpg.object_crash_vehicle import DynamicObjectCrossing as scenario_03_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.object_crash_intersection import VehicleTurningRoute as scenario_04_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.other_leading_vehicle import OtherLeadingVehicle as scenario_05_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_advpolicy
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_advpolicy


SECONDS_GIVEN_PER_METERS = 1
SCENARIO_CLASS_MAPPING = {
    "ordinary": {
        "Scenario0": AutopolitBackgroundVehicle,
    },
    "standard": {
        "Scenario3": scenario_03_standard,
        "Scenario4": scenario_04_standard,
        "Scenario5": scenario_05_standard,
        "Scenario6": scenario_06_standard,
        "Scenario7": scenario_07_standard,
        "Scenario8": scenario_08_standard,
        "Scenario9": scenario_09_standard,
        "Scenario10": scenario_10_standard,
    },
    'benign': {
        "Scenario3": scenario_03_benign,
        "Scenario4": scenario_04_benign,
        "Scenario5": scenario_05_benign,
        "Scenario6": scenario_06_benign,
        "Scenario7": scenario_07_benign,
        "Scenario8": scenario_08_benign,
        "Scenario9": scenario_09_benign,
        "Scenario10": scenario_10_benign,
    },
    'carla_challenge': {
        "Scenario3": scenario_03_carla_challenge,
        "Scenario4": scenario_04_carla_challenge,
        "Scenario5": scenario_05_carla_challenge,
        "Scenario6": scenario_06_carla_challenge,
        "Scenario7": scenario_07_carla_challenge,
        "Scenario8": scenario_08_carla_challenge,
        "Scenario9": scenario_09_carla_challenge,
        "Scenario10": scenario_10_carla_challenge,
    },
    'LC': {
        "Scenario3": scenario_03_lc,
        "Scenario4": scenario_04_lc,
        "Scenario5": scenario_05_lc,
        "Scenario6": scenario_06_lc,
        "Scenario7": scenario_07_lc,
        "Scenario8": scenario_08_lc,
        "Scenario9": scenario_09_lc,
        "Scenario10": scenario_10_lc,
    },
    'advtraj': {
        "Scenario3": scenario_03_advtraj,
        "Scenario4": scenario_04_advtraj,
        "Scenario5": scenario_05_advtraj,
        "Scenario6": scenario_06_advtraj,
        "Scenario7": scenario_07_advtraj,
        "Scenario8": scenario_08_advtraj,
        "Scenario9": scenario_09_advtraj,
        "Scenario10": scenario_10_advtraj,
    },
    'advsim': {
        "Scenario3": scenario_03_advsim,
        "Scenario4": scenario_04_advsim,
        "Scenario5": scenario_05_advsim,
        "Scenario6": scenario_06_advsim,
        "Scenario7": scenario_07_advsim,
        "Scenario8": scenario_08_advsim,
        "Scenario9": scenario_09_advsim,
        "Scenario10": scenario_10_advsim,
    },
    'advmaddpg': {
        "Scenario3": scenario_03_advpolicy,
        "Scenario4": scenario_04_advpolicy,
        "Scenario5": scenario_05_advpolicy,
        "Scenario6": scenario_06_advpolicy,
        "Scenario7": scenario_07_advpolicy,
        "Scenario8": scenario_08_advpolicy,
        "Scenario9": scenario_09_advpolicy,
        "Scenario10": scenario_10_advpolicy,
    },
}


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60

        self.route, self.ego_vehicle, scenario_definitions = self._update_route_and_ego()
        self.other_actors = []
        self.list_scenarios = self._build_scenario_instances(scenario_definitions)
        self.criteria = self._create_criteria()

    def _update_route_and_ego(self, timeout=None):
        # transform the scenario file into a dictionary
        if self.config.scenario_file is not None:
            world_annotations = RouteParser.parse_annotations_file(self.config.scenario_file)
        else:
            world_annotations = self.config.scenario_config

            # prepare route's trajectory (interpolate and add the GPS route)
        ego_vehicle = None
        route = None
        scenario_id = self.config.scenario_id
        if scenario_id == 0:
            vehicle_spawn_points = get_valid_spawn_points(self.world)
            for random_transform in vehicle_spawn_points:
                route = interpolate_trajectory(self.world, [random_transform])
                ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)
                if ego_vehicle is not None:
                    break
        else:
            route = interpolate_trajectory(self.world, self.config.trajectory)
            ego_vehicle = self._spawn_ego_vehicle(route[0][0], self.config.auto_ego)
        
        # scan route to get exactly 1 scenario definition
        possible_scenarios, _ = RouteParser.scan_route_for_scenarios(
            self.config.town,
            route,
            world_annotations,
            scenario_id=scenario_id
        )
        
        scenarios_definitions = []
        for trigger in possible_scenarios.keys():
            scenarios_definitions.extend(possible_scenarios[trigger])

        assert len(scenarios_definitions) == 1, "There should be exactly 1 scenario definition in the route"

        # TODO: ego route will be overwritten by other scenarios
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle, scenarios_definitions

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform, autopilot=False):
        try:
            role_name = 'ego_vehicle' + str(self.ego_id)
            ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', elevate_transform, rolename=role_name, autopilot=autopilot)
            ego_vehicle.set_autopilot(autopilot, CarlaDataProvider.get_traffic_manager_port())
        except Exception as e:
            raise RuntimeError("Error while spawning ego vehicle: {}".format(e))
        return ego_vehicle

    def _build_scenario_instances(self, scenario_definitions):
        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_list = []
        for scenario_number, definition in enumerate(scenario_definitions):
            # get the class possibilities for this scenario number
            scenario_class = SCENARIO_CLASS_MAPPING[self.config.scenario_generation_method][definition['name']]

            # create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []

            # create an actor configuration for the ego-vehicle trigger position
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            route_config = RouteScenarioConfig()
            route_config.other_actors = list_of_actor_conf_instances
            route_config.trigger_points = [egoactor_trigger_position]
            #route_config.subtype = definition['scenario_type']
            route_config.parameters = self.config.parameters
            route_config.num_scenario = self.config.num_scenario
            if self.config.weather is not None:
                route_config.weather = self.config.weather

            try:
                scenario_instance = scenario_class(self.world, self.ego_vehicle, route_config, timeout=self.timeout)
            except Exception as e:   
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_list.append(scenario_instance)
        return scenario_instance_list

    def _get_actors_instances(self, list_of_antagonist_actors):
        def get_actors_from_list(list_of_actor_def):
            # receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors

        list_of_actors = []
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])
        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])
        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        return list_of_actors

    def initialize_actors(self):
        amount = 0 
        new_actors = CarlaDataProvider.request_new_batch_actors(
            'vehicle.*', 
            amount, 
            carla.Transform(), 
            autopilot=True, 
            random_location=True, 
            rolename='background'
        )
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        for _actor in new_actors:
            self.other_actors.append(_actor)

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_acceleration_x': self.ego_vehicle.get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicle.get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicle.get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicle).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicle).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicle).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Stop due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Stop due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Stop due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Stop due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            # only check when evaluating
            if self.config.scenario_id != 0:  
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Stop due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Stop due to timeout', color='yellow') 
                break

        return running_status, stop

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()

        # each scenario remove its own actors
        for scenario in self.list_scenarios:
            scenario.clean_up()

        # remove background vehicles
        for s_i in range(len(self.other_actors)):
            if self.other_actors[s_i].type_id.startswith('vehicle'):
                self.other_actors[s_i].set_autopilot(enabled=False)
            if CarlaDataProvider.actor_id_exists(self.other_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.other_actors[s_i].id)
        self.other_actors = []
