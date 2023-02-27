'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-27 16:45:21
Description: 
'''

import math
import traceback

from numpy import random
import carla
import xml.etree.ElementTree as ET

from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider

from safebench.scenario.scenario_configs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
from safebench.scenario.tools.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

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
from safebench.scenario.scenario_definition.standard.object_crash_vehicle_dynamic import DynamicObjectCrossing as scenario_03_standard
from safebench.scenario.scenario_definition.standard.object_crash_intersection_dynamic import VehicleTurningRoute as scenario_04_standard
from safebench.scenario.scenario_definition.standard.other_leading_vehicle_dynamic import OtherLeadingVehicle as scenario_05_standard
from safebench.scenario.scenario_definition.standard.maneuver_opposite_direction_dynamic import ManeuverOppositeDirection as scenario_06_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route_dynamic import OppositeVehicleRunningRedLight as scenario_07_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route_dynamic import SignalizedJunctionLeftTurn as scenario_08_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route_dynamic import SignalizedJunctionRightTurn as scenario_09_standard
from safebench.scenario.scenario_definition.standard.junction_crossing_route_dynamic import NoSignalJunctionCrossingRoute as scenario_10_standard

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

# AdvMADDGP
from safebench.scenario.scenario_definition.advmaddpg.object_crash_vehicle import DynamicObjectCrossing as scenario_03_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.object_crash_intersection import VehicleTurningRoute as scenario_04_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.other_leading_vehicle import OtherLeadingVehicle as scenario_05_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.maneuver_opposite_direction import ManeuverOppositeDirection as scenario_06_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import OppositeVehicleRunningRedLight as scenario_07_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import SignalizedJunctionLeftTurn as scenario_08_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import SignalizedJunctionRightTurn as scenario_09_advmaddpg
from safebench.scenario.scenario_definition.advmaddpg.junction_crossing_route import NoSignalJunctionCrossingRoute as scenario_10_advmaddpg


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
        "Scenario3": scenario_03_advmaddpg,
        "Scenario4": scenario_04_advmaddpg,
        "Scenario5": scenario_05_advmaddpg,
        "Scenario6": scenario_06_advmaddpg,
        "Scenario7": scenario_07_advmaddpg,
        "Scenario8": scenario_08_advmaddpg,
        "Scenario9": scenario_09_advmaddpg,
        "Scenario10": scenario_10_advmaddpg,
    },
}


def convert_json_to_transform(actor_dict):
    """
        Convert a JSON string to a CARLA transform
    """
    return carla.Transform(
        location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']), z=float(actor_dict['z'])),
        rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw']))
    )


def convert_json_to_actor(actor_dict):
    """
        Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
        Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
        Compare function for scenarios based on distance of the scenario start position
    """

    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']
        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare
    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True
    return False


class RouteScenario():
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger

        self.config = config
        self.route = None
        self.ego_id = ego_id
        self.sampled_scenarios_definitions = None
        self.max_running_step = max_running_step
        self.timeout = 60

        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # self._update_route(world, config)
        # ego_vehicle = self._update_ego_vehicle()
        ego_vehicle = self._update_route_and_ego(world, config)
        self.ego_vehicles = [ego_vehicle]
        self.ego_idx = 0
        self.other_actors = []

        self.list_scenarios = self._build_scenario_instances(
            world,
            ego_vehicle,
            self.sampled_scenarios_definitions,
            scenarios_per_tick=5,
            timeout=self.timeout,
            weather=config.weather
        )
        self.criteria = self._create_criteria()

    def _update_route_and_ego(self, world, config, timeout=None):
        # Transform the scenario file into a dictionary
        if config.scenario_file is not None:
            world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        else:
            world_annotations = config.scenario_config

        # prepare route's trajectory (interpolate and add the GPS route)
        ego_vehicle = None
        if self.config.scenario_id == 0:
            vehicle_spawn_points = self.world.get_map().get_spawn_points()
            random.shuffle(vehicle_spawn_points)
            for random_transform in vehicle_spawn_points:
                gps_route, route = interpolate_trajectory(world, [random_transform])
                ego_vehicle = self._spawn_ego_vehicle(route[0][0])
                if ego_vehicle is not None:
                    break
        else:
            gps_route, route = interpolate_trajectory(world, config.trajectory)
            ego_vehicle = self._spawn_ego_vehicle(route[0][0])

        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations, scenario_id=self.config.scenario_id)
        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        CarlaDataProvider.set_scenario_config(config)

        if config.agent is not None:
            config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout() if timeout is None else timeout
        return ego_vehicle

    def _scenario_sampling(self, potential_scenarios_definitions):
        """
            The function used to sample the scenarios that are going to happen for this route.
        """
        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True
            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = random.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = random.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)
        return sampled_scenarios

    def _estimate_route_timeout(self):
        route_length = 0.0  # in meters
        min_length = 100.0

        if len(self.route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _spawn_ego_vehicle(self, elevate_transform):
        # gradually increase the height of ego vehicle
        success = False
        start_z = elevate_transform.location.z
        while not success:
            try:
                if elevate_transform.location.z - start_z > 0.5:
                    return None
                role_name = 'ego_vehicle' + str(self.ego_id)
                ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', elevate_transform, rolename=role_name)
                if ego_vehicle is not None:
                    success = True
                else:
                    elevate_transform.location.z += 0.1
            except RuntimeError:
                elevate_transform.location.z += 0.1
        return ego_vehicle

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, weather=None):
        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []
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
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.parameters = self.config.parameters
            scenario_configuration.num_scenario = self.config.num_scenario

            if weather is not None:
                scenario_configuration.weather = weather

            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.tesla.model3', ego_vehicle.get_transform(), 'ego_vehicle')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name

            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration, timeout=timeout)
                # tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                scenario_number += 1
            except Exception as e:   
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)
        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        def get_actors_from_list(list_of_actor_def):
            # receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))
            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])
        return list_of_actors

    def initialize_actors(self):
        """
            Set other_actors to the superset of all scenario actors
        """
        # TODO: totally remove background vehicle
        if self.config.initialize_background_actors:
            amount = 0 
        else:
            amount = 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*', amount, carla.Transform(), autopilot=True, random_location=True, rolename='background')
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")
        
        for _actor in new_actors:
            print('background', _actor.type_id)
            self.other_actors.append(_actor)

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicles[self.ego_idx]),
            'ego_acceleration_x': self.ego_vehicles[self.ego_idx].get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicles[self.ego_idx].get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicles[self.ego_idx].get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicles[self.ego_idx]).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Stop due to collision', color='yellow')
        if self.config.scenario_id != 0:  # only check when evaluating
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Stop due to route completion', color='yellow')
            if running_status['speed_above_threshold'] == Status.FAILURE:
                if running_status['route_complete'] == 0:
                    raise RuntimeError("Agent not moving")
                else:
                    stop = True
                    self.logger.log('>> Stop due to low speed', color='yellow')
        else:
            if len(running_record) >= self.max_running_step:  # stop at max step when training
                stop = True
                self.logger.log('>> Stop due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            # print(running_status['driven_distance'])
            if self.config.scenario_id != 0:  # only check when evaluating
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

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicles[self.ego_idx], distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicles[self.ego_idx], avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicles[self.ego_idx], optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicles[self.ego_idx], optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicles[self.ego_idx], terminate_on_failure=True)
        # criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicles[self.ego_idx])
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicles[self.ego_idx])
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicles[self.ego_idx], route=route, offroad_max=30)
            criteria['speed_above_threshold'] = ActorSpeedAboveThresholdTest(
                actor=self.ego_vehicles[self.ego_idx],
                speed_threshold=0.1,
                below_threshold_max_time=10,
                terminate_on_failure=True
            )
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicles[self.ego_idx], route=route)
        return criteria

    def clean_up(self):
        # stop criterion
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
