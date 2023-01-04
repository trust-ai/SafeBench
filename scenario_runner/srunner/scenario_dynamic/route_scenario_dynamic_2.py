"""
@author: Shuai Wang
@e-mail: ws199807@outlook.com
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import traceback
import xml.etree.ElementTree as ET
from numpy import random

import py_trees

import carla

from agents.navigation.local_planner import RoadOption

from scenario_runner.srunner.scenario_manager.scenarioatomics.atomic_criteria import (Status,
                                                                     CollisionTest,
                                                                     DrivenDistanceTest,
                                                                     AverageVelocityTest,
                                                                     OffRoadTest,
                                                                     KeepLaneTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)
from scenario_runner.srunner.scenario_manager.timer import GameTime
from scenario_runner.srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from scenario_runner.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
# from scenario_runner.srunner.scenario_manager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from scenario_runner.srunner.scenario_dynamic.basic_scenario_dynamic import BasicScenarioDynamic
from scenario_runner.srunner.tools.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from scenario_runner.srunner.tools.route_manipulation import interpolate_trajectory


# standard
from scenario_runner.srunner.scenario_dynamic.standard.object_crash_vehicle_dynamic import DynamicObjectCrossingDynamic
from scenario_runner.srunner.scenario_dynamic.standard.object_crash_intersection_dynamic import VehicleTurningRouteDynamic
from scenario_runner.srunner.scenario_dynamic.standard.other_leading_vehicle_dynamic import OtherLeadingVehicleDynamic
from scenario_runner.srunner.scenario_dynamic.standard.maneuver_opposite_direction_dynamic import ManeuverOppositeDirectionDynamic
from scenario_runner.srunner.scenario_dynamic.standard.junction_crossing_route_dynamic import OppositeVehicleRunningRedLightDynamic
from scenario_runner.srunner.scenario_dynamic.standard.junction_crossing_route_dynamic import SignalizedJunctionLeftTurnDynamic
from scenario_runner.srunner.scenario_dynamic.standard.junction_crossing_route_dynamic import SignalizedJunctionRightTurnDynamic
from scenario_runner.srunner.scenario_dynamic.standard.junction_crossing_route_dynamic import NoSignalJunctionCrossingRouteDynamic

# lc
# from srunner.scenario_dynamic.LC.object_crash_vehicle import DynamicObjectCrossingDynamic as scenario_03_lc
# from srunner.scenario_dynamic.LC.object_crash_intersection import VehicleTurningRouteDynamic as scenario_04_lc
# from srunner.scenario_dynamic.LC.other_leading_vehicle import OtherLeadingVehicleDynamic as scenario_05_lc
# from srunner.scenario_dynamic.LC.maneuver_opposite_direction import ManeuverOppositeDirectionDynamic as scenario_06_lc
from scenario_runner.srunner.scenario_dynamic.LC.junction_crossing_route import OppositeVehicleRunningRedLightDynamic as scenario_07_lc
# from srunner.scenario_dynamic.LC.junction_crossing_route import SignalizedJunctionLeftTurnDynamic as scenario_08_lc
# from srunner.scenario_dynamic.LC.junction_crossing_route import SignalizedJunctionRightTurnDynamic as scenario_09_lc
# from srunner.scenario_dynamic.LC.junction_crossing_route import NoSignalJunctionCrossingRouteDynamic as scenario_10_lc

# carla_challenge

# from scenario_runner.srunner.scenario_dynamic.junction_crossing_route_dynamic import OppositeVehicleRunningRedLightDynamic
# from srunner.scenario_dynamic.carla_challenge.object_crash_vehicle import DynamicObjectCrossingDynamic as scenario_03_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.object_crash_intersection import VehicleTurningRouteDynamic as scenario_04_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.other_leading_vehicle import OtherLeadingVehicleDynamic as scenario_05_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.maneuver_opposite_direction import ManeuverOppositeDirectionDynamic as scenario_06_carla_challenge
from scenario_runner.srunner.scenario_dynamic.carla_challenge.junction_crossing_route import OppositeVehicleRunningRedLightDynamic as scenario_07_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.junction_crossing_route import SignalizedJunctionLeftTurnDynamic as scenario_08_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.junction_crossing_route import SignalizedJunctionRightTurnDynamic as scenario_09_carla_challenge
# from srunner.scenario_dynamic.carla_challenge.junction_crossing_route import NoSignalJunctionCrossingRouteDynamic as scenario_10_carla_challenge


SECONDS_GIVEN_PER_METERS = 1

NUMBER_CLASS_TRANSLATION = {
    "standard": {
        "Scenario3": OtherLeadingVehicleDynamic,
        "Scenario4": OtherLeadingVehicleDynamic,
        "Scenario5": OtherLeadingVehicleDynamic,
        "Scenario6": OtherLeadingVehicleDynamic,
        "Scenario7": OtherLeadingVehicleDynamic,
        "Scenario8": OtherLeadingVehicleDynamic,
        "Scenario9": OtherLeadingVehicleDynamic,
        "Scenario10": OtherLeadingVehicleDynamic,
    },
    'carla_challenge': {
        "Scenario3": scenario_07_carla_challenge,
        "Scenario4": scenario_07_carla_challenge,
        "Scenario5": scenario_07_carla_challenge,
        "Scenario6": scenario_07_carla_challenge,
        "Scenario7": scenario_07_carla_challenge,
        "Scenario8": scenario_07_carla_challenge,
        "Scenario9": scenario_07_carla_challenge,
        "Scenario10": scenario_07_carla_challenge,
    },
    'lc': {
        "Scenario3": scenario_07_lc,
        "Scenario4": scenario_07_lc,
        "Scenario5": scenario_07_lc,
        "Scenario6": scenario_07_lc,
        "Scenario7": scenario_07_lc,
        "Scenario8": scenario_07_lc,
        "Scenario9": scenario_07_lc,
        "Scenario10": scenario_07_lc,
    },
}


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


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


class RouteScenarioDynamic(BasicScenarioDynamic):
    """
        Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
        along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, ego_id,debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.world = world
        self.config = config
        self.route = None
        self.ego_id = ego_id
        self.sampled_scenarios_definitions = None

        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        self._update_route(world, config, debug_mode)

        ego_vehicle = self._update_ego_vehicle()


        self.list_scenarios = self._build_scenario_instances(world,
                                                             ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=5,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode,
                                                             weather=config.weather)

        # print("list_scenarios: ", self.list_scenarios)

        super(RouteScenarioDynamic, self).__init__(name=config.name,
                                                   ego_vehicles=[ego_vehicle],
                                                   config=config,
                                                   world=world,
                                                   debug_mode=False,
                                                   terminate_on_failure=False,
                                                   criteria_enable=criteria_enable)

        self.criteria = self._create_criteria()

    def _update_route(self, world, config, debug_mode, timeout=None):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        if config.scenario_file is not None:
            world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        else:
            world_annotations = config.scenario_config

        # prepare route's trajectory (interpolate and add the GPS route)
        len_trajectory = len(config.trajectory)
        # print(f"length of trajectory {len_trajectory}")
        if len_trajectory == 0:
            len_spawn_points = len(self.vehicle_spawn_points)
            idx = random.choice(list(range(len_spawn_points)))
            print('choosing spawn point {} from {} points'.format(idx, len_spawn_points))
            random_transform = self.vehicle_spawn_points[idx]
            gps_route, route = interpolate_trajectory(world, [random_transform])
        else:
            gps_route, route = interpolate_trajectory(world, config.trajectory)

        potential_scenarios_definitions, _, t, mt = RouteParser.scan_route_for_scenarios(config.town, route,
                                                                                         world_annotations)
        print('matched_triggers', mt)
        print('scenarios', potential_scenarios_definitions)

        self.route = route
        self.route_length = len(route)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        CarlaDataProvider.set_scenario_config(config)
        print('ego route updated!')

        if config.agent is not None:
            config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout() if timeout is None else timeout

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rng = random.RandomState(random_seed)

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

            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
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

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        # elevate_transform.location.z += 0.5

        print(" =================================== elevate transform")
        print(elevate_transform)
        success = False
        # NOTE: request actor has bug
        while not success:
            print(success)
            try:
                role_name = 'ego_vehicle'+str(self.ego_id)
                print(role_name)
                ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
                                                                  elevate_transform,
                                                                  rolename='ego_vehicle'+str(self.ego_id),
                                                                  tick=False)
                success = True
            except RuntimeError:
                elevate_transform.location.z += 0.1


        # Collision sensor
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Lidar sensor
        lidar_height = 2.1
        lidar_trans = carla.Transform(carla.Location(x=0.0, z=lidar_height))
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '5000')

        # Camera sensor
        camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        # Set the time in seconds between sensor captures
        camera_bp.set_attribute('sensor_tick', '0.05')

        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego_vehicle)
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_trans, attach_to=ego_vehicle)
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_trans, attach_to=ego_vehicle)

        print("========= ego vehicle: ")
        print(ego_vehicle)
        return ego_vehicle

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False, weather=None):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            scenario_class = NUMBER_CLASS_TRANSLATION[self.config.scenario_generation_method][definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.parameters = self.config.parameters

            if weather is not None:
                scenario_configuration.weather = weather

            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          ego_vehicle.get_transform(),
                                                                          'ego_vehicle')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name

            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                   criteria_enable=False, timeout=timeout)
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                scenario_number += 1
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
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
        config = self.config
        # Create the background activity of the route
        town_amount = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120,
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10': 120,
        }

        if config.initialize_background_actors:
            amount = town_amount[config.town] if config.town in town_amount else 0
        else:
            amount = 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def update_behavior(self):
        """
        This route scenario doesn't define updating rules for actors in small scenarios
        """
        pass

    def get_running_status(self, running_record):
        running_status = {'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicles[0]),
                          'ego_acceleration_x': self.ego_vehicles[0].get_acceleration().x,
                          'ego_acceleration_y': self.ego_vehicles[0].get_acceleration().y,
                          'ego_acceleration_z': self.ego_vehicles[0].get_acceleration().z,
                          'ego_x': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.x,
                          'ego_y': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.y,
                          'ego_z': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.z,
                          'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.roll,
                          'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.pitch,
                          'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.yaw,
                          'current_game_time': GameTime.get_time()}

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        # print("running status: ", running_status)

        # print(running_status['ego_velocity'])
        # print(running_status['route_complete'])

        stop = False
        if running_status['collision'] == Status.FAILURE:
            stop = True
            print('stop due to collision')
        if self.route_length > 1:  # only check when evaluating
            # print(running_status['route_complete'])
            if running_status['route_complete'] == 100:
                stop = True
                print('stop due to route completion')
            if running_status['speed_above_threshold'] == Status.FAILURE:
                if running_status['route_complete'] == 0:
                    raise RuntimeError("Agent not moving")
                else:
                    stop = True
                    print('stop due to low speed')
        else:
            if len(running_record) >= 250:  # stop at max step when training
                stop = True
                print('stop due to max steps')

        for scenario in self.list_scenarios:
            # print(running_status['driven_distance'])
            if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                stop = True
                print('stop due to max driven distance')
                break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                print('stop due to timeout')
                break

        return running_status, stop

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        We need this method for background
        We keep pytrees just for background
        """
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                   policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        # subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)

        return behavior

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicles[0], distance_success=1e4,
                                                         distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicles[0], avg_velocity_success=1e4,
                                                           avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicles[0], optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicles[0], optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicles[0], terminate_on_failure=True)
        # criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicles[0])
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicles[0])
        if self.route_length > 1:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30)
            criteria['speed_above_threshold'] = ActorSpeedAboveThresholdTest(actor=self.ego_vehicles[0],
                                                                             speed_threshold=0.1,
                                                                             below_threshold_max_time=10,
                                                                             terminate_on_failure=True)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicles[0], route=route)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        for criterion_name, criterion in self.criteria.items():
            criterion.terminate()
        for scenario in self.list_scenarios:
            scenario.remove_all_actors()
        self.remove_all_actors()

