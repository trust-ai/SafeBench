#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA scenario_runner

This is the main script to be executed when running a scenario.
It loads the scenario configuration, loads the scenario and manager,
and finally triggers the scenario execution.
"""

from __future__ import print_function

import glob
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import signal
import sys
import time
import json
import pkg_resources
import random
import numpy as np
import torch
import rospy
import joblib
from enum import Enum

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager_dynamic import ScenarioManagerDynamic
from srunner.scenario_dynamic.route_scenario_dynamic import RouteScenarioDynamic
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

from geometry_msgs.msg import PoseWithCovarianceStamped
from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus
from carla_ros_scenario_runner_types.srv import GetEgoVehicleRoute
from carla_ros_scenario_runner_types.srv import UpdateRenderMap

# Version of scenario_runner
VERSION = '0.9.11'


class ApplicationStatus(Enum):
    """
    States of an application
    """
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    SHUTTINGDOWN = 3
    ERROR = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScenarioRunner(object):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    scenario_runner = ScenarioRunner(args)
    scenario_runner.run()
    del scenario_runner
    """
    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self._args = args

        if args.timeout:
            self.client_timeout = float(args.timeout)

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(self._args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.10'):
            raise ImportError("CARLA version 0.9.10 or newer required. CARLA version found: {}".format(dist))

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self._args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        # comment: Fred 12.3
        self.manager = ScenarioManagerDynamic(self._args.debug, self._args.sync, self._args.timeout)

        # Create signal handler for SIGINT
        self._shutdown_requested = False
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()

        self.scenario_status_publisher = rospy.Publisher("/scenario/status", CarlaScenarioStatus,
                                                         latch=True, queue_size=1)
        # self.initial_pose_publisher = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)

    def scenario_status_updated(self, status):
        rospy.loginfo("Scenario status updated to {}".format(status))
        val = CarlaScenarioStatus.STOPPED
        if status == ApplicationStatus.STOPPED:
            val = CarlaScenarioStatus.STOPPED
        elif status == ApplicationStatus.STARTING:
            val = CarlaScenarioStatus.STARTING
        elif status == ApplicationStatus.RUNNING:
            val = CarlaScenarioStatus.RUNNING
        elif status == ApplicationStatus.SHUTTINGDOWN:
            val = CarlaScenarioStatus.SHUTTINGDOWN
        else:
            val = CarlaScenarioStatus.ERROR
        status = CarlaScenarioStatus()
        status.status = val
        self.scenario_status_publisher.publish(status)

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()
            self._cleanup()
            if not self.manager.get_running_status():
                raise RuntimeError("Timeout occured during scenario execution")

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:

            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Remove unused Python paths
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        self.scenario_status_updated(ApplicationStatus.STOPPED)

        # Simulation still running and in synchronous mode?
        if self.world is not None and self._args.sync:
            try:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except RuntimeError:
                sys.exit(-1)

        self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self._args.waitForEgo:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        self._clear_all_actors([
            'sensor.other.collision', 'sensor.lidar.ray_cast',
            'sensor.other.lane_invasion', 'sensor.camera.rgb',
            # 'sensor.pseudo.tf', 'sensor.pseudo.objects',
            # 'sensor.pseudo.odom', 'sensor.pseudo.speedometer',
            'vehicle.*', 'controller.ai.walker',
            'walker.*'
        ])

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or actor.type_id.startswith('sensor'):
                        actor.stop()
                    actor.destroy()

    def _find_default_ego_vehicles(self):
        ego_vehicles = []
        for actor in self.client.get_world().get_actors():
            if actor.attributes.get('role_name') == 'ego_vehicle':
                ego_vehicles.append(actor)
        self.ego_vehicles = ego_vehicles
        return ego_vehicles

    # def _set_initial_position(self, initial_pose):
    #     init_pose = PoseWithCovarianceStamped()
    #     init_pose.header.stamp = rospy.get_rostime()
    #     init_pose.header.frame_id = "map"
    #     init_pose.pose.pose = initial_pose
    #     rospy.loginfo('set initial pose (ROS) for ego: ({}, {})'.format(initial_pose.position.x, initial_pose.position.y))
    #     self.initial_pose_publisher.publish(init_pose)

    def _prepare_ego_vehicles(self, ego_vehicles):
        """
        Spawn or update the ego vehicles
        """

        if not self._args.waitForEgo:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                CarlaDataProvider.register_actor(self.ego_vehicles[i])

        # sync state
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _analyze_scenario(self, config):
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        json_filename = None
        config_name = config.name
        if self._args.outputDir != '':
            config_name = os.path.join(self._args.outputDir, config_name)

        if self._args.junit:
            junit_filename = config_name + current_time + ".xml"
        if self._args.json:
            json_filename = config_name + current_time + ".json"
        filename = None
        if self._args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self._args.output, filename, junit_filename, json_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self._args.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _record_criteria(self, criteria, name):
        """
        Filter the JSON serializable attributes of the criterias and
        dumps them into a file. This will be used by the metrics manager,
        in case the user wants specific information about the criterias.
        """
        file_name = name[:-4] + ".json"

        # Filter the attributes that aren't JSON serializable
        with open('temp.json', 'w') as fp:

            criteria_dict = {}
            for criterion in criteria:

                criterion_dict = criterion.__dict__
                criteria_dict[criterion.name] = {}

                for key in criterion_dict:
                    if key != "name":
                        try:
                            key_dict = {key: criterion_dict[key]}
                            json.dump(key_dict, fp, sort_keys=False, indent=4)
                            criteria_dict[criterion.name].update(key_dict)
                        except TypeError:
                            pass

        os.remove('temp.json')

        # Save the criteria dictionary into a .json file
        with open(file_name, 'w') as fp:
            json.dump(criteria_dict, fp, sort_keys=False, indent=4)

    def _load_and_wait_for_world(self, town, config=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        ego_vehicles = None if config is None else config.ego_vehicles
        print("============= load and wait for world")

        # print(self.client.get_world().get_map().name)
        # self.world = self.client.load_world('Town02')
        # print(self.client.get_world().get_map().name)
        # while True:
        #     pass

        if self._args.reloadWorld:
            print("============= world reloading")
            self.world = self.client.load_world(town)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            previous_town = self.client.get_world().get_map().name
            updated = previous_town == town
            if not updated:
                print('loading map', town)
                self.world = self.client.load_world(town)

            while not updated:
                response = None
                rospy.wait_for_service('/gym_node/update_render_map')
                try:
                    requester = rospy.ServiceProxy('/gym_node/update_render_map', UpdateRenderMap)
                    print('updating render map')
                    response = requester(town)
                    if response is not None:
                        updated = response.result
                except rospy.ServiceException as e:
                    rospy.loginfo('Run scenario service call failed: {}'.format(e))

            ego_vehicle_found = False
            # if config.initial_transform is not None:
            #     # TODO: check initial pose
            #     self._set_initial_position(config.initial_transform)
            print("============= waiting for ego vehicle")
            if self._args.waitForEgo:
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    print(ego_vehicles, vehicles, config, config.ego_vehicles, self.ego_vehicles)
                    if len(ego_vehicles) == 0:
                        print("============= Empty ego vehicle list, checking default rolename 'ego_vehicle'")
                        default_rolename_found = False
                        for actor in self.client.get_world().get_actors():
                            if actor.attributes.get('role_name') == 'ego_vehicle':
                                default_rolename_found = True
                        if default_rolename_found:
                            print("============= Found default rolename 'ego_vehicle'. Stop waiting")
                            break
                        else:
                            print("============= Default rolename 'ego_vehicle' not found. Waiting ... ")
                    for ego_vehicle in ego_vehicles:
                        print("============= Enter here")
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            # if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            if vehicle.attributes['role_name'] == ego_vehicle.attributes['role_name']:
                                ego_vehicle_found = True
                                if config.initial_transform is not None:
                                # if config.initial_pose is not None:
                                    before = ego_vehicle.get_transform()
                                    print('Setting ego transform', config.initial_transform)
                                    location_correct = False
                                    num_trial = 0
                                    world = self.client.get_world()
                                    while not location_correct:
                                        num_trial += 1
                                        ego_vehicle.set_transform(config.initial_transform)
                                        # self._set_initial_position(config.initial_pose)
                                        time.sleep(num_trial)
                                        world.wait_for_tick()
                                        current_transform = ego_vehicle.get_transform()
                                        distance = config.initial_transform.location.distance(current_transform.location)
                                        print(current_transform.location, config.initial_transform.location, distance)
                                        if distance <= 2.5:
                                            location_correct = True
                                        if num_trial >= 10 and not location_correct:
                                            return False
                                    after = ego_vehicle.get_transform()
                                    print('trial', num_trial, 'before', before, 'after', after)
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()

        # if self._args.sync:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(self._args.trafficManagerSeed))

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.tick = self.world.tick()
        else:
            self.world.wait_for_tick()
        if CarlaDataProvider.get_map().name != town and CarlaDataProvider.get_map().name != "OpenDriveMap":
            print("The CARLA server uses the wrong map: {}".format(CarlaDataProvider.get_map().name))
            print("This scenario requires to use map: {}".format(town))
            return False
        print("============= finished world loading")

        return True

    def _load_and_run_scenario(self, config):
        """
        Load and run the scenario given by config
        """
        self.scenario_status_updated(ApplicationStatus.STARTING)
        result = False
        record = {}
        if not self._load_and_wait_for_world(config.town, config):
            self._cleanup()
            return False, record

        if self._args.agent:
            print("============= create agent")
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False, record

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            print("============= create route scenario")

            # TODO: visualize scenario (10.12)

            scenario = None
            scenario = RouteScenarioDynamic(world=self.world, config=config, debug_mode=self._args.debug, timeout=800)

        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False, record

        try:
            if self._args.record:
                recorder_dir = self._args.record
                # recorder_dir = os.path.join(os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record)
                recorder_name = os.path.join(recorder_dir, '{}.log'.format(config.name))
                print('creating dir {}'.format(recorder_dir))
                os.makedirs(recorder_dir, exist_ok=True)
                print('logging record to {}'.format(recorder_name))
                # recorder_name = "{}/{}/{}.log".format(
                #     os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                self.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)
            self.scenario_status_updated(ApplicationStatus.RUNNING)
            record = self.manager.run_scenario()
            self.scenario_status_updated(ApplicationStatus.SHUTTINGDOWN)

            # Provide outputs if required
            self._analyze_scenario(config)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)

            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result, record

    def _run_scenarios(self):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False

        # Load the scenario configurations provided in the config file
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
            self._args.scenario,
            self._args.configFile)
        if not scenario_configurations:
            print("Configuration for scenario {} cannot be found!".format(self._args.scenario))
            return result

        # Execute each configuration
        for config in scenario_configurations:
            for _ in range(self._args.repetitions):
                result = self._load_and_run_scenario(config)

            self._cleanup()
        return result

    def _run_route(self):
        """
        Run the route scenario
        """
        result = False

        print('Using data file:', self._args.data_file)
        with open(self._args.data_file, 'r') as f:
            data_full = json.loads(f.read())
            if self._args.method is not None:
                print('selecting method:', self._args.method)
                data_full = [item for item in data_full if item["method"] == self._args.method]
            if self._args.scenario_id is not None:
                print('selecting scenario_id:', self._args.scenario_id)
                data_full = [item for item in data_full if item["scenario_id"] == self._args.scenario_id]
            if self._args.route_id is not None:
                print('selecting route_id:', self._args.route_id)
                data_full = [item for item in data_full if item["route_id"] == self._args.route_id]
            if self._args.risk_level is not None:
                print('selecting risk_level:', self._args.risk_level)
                data_full = [item for item in data_full if item["risk_level"] == self._args.risk_level]
        print('loading {} data'.format(len(data_full)))

        route_configurations = []
        route_file_formatter = '/home/carla/Evaluation/src/evaluation/scenario_node/route/scenario_%02d_routes/scenario_%02d_route_%02d.xml'
        scenario_file_formatter = '/home/carla/Evaluation/src/evaluation/scenario_node/route/scenarios/scenario_%02d.json'
        # testing_method = None
        for item in data_full:
            route_file = route_file_formatter % (item['scenario_id'], item['scenario_id'], item['route_id'])
            scenario_file = scenario_file_formatter % item['scenario_id']
            parsed_configs = RouteParser.parse_routes_file(route_file, scenario_file)
            assert len(parsed_configs) == 1, item
            config = parsed_configs[0]
            config.data_id = item['data_id']
            config.scenario_generation_method = item['method']
            config.scenario_id = item['scenario_id']
            config.route_id = item['route_id']
            config.risk_level = item['risk_level']
            config.parameters = item['parameters']

            route_configurations.append(config)

            # if testing_method is not None:
            #     assert testing_method == item['method']
            # else:
            #     testing_method = item['method']

        if self._args.train_agent:
            print('training agent...')
            for episode in range(self._args.train_agent_episodes):
                print('episode {}/{}'.format(episode + 1, self._args.train_agent_episodes))
                for config_idx, config in enumerate(route_configurations):
                    print('config {}/{}'.format(config_idx + 1, len(route_configurations)))
                    result, record = self._load_and_run_scenario(config)
                    self._cleanup()
        else:
            record_dir = os.path.join(self._args.outputDir, 'testing_records')
            os.makedirs(record_dir, exist_ok=True)
            # record_filename = os.path.join(record_dir, '{}.pkl'.format(testing_method))
            record_filename = os.path.join(record_dir, 'record.pkl')
            if os.path.exists(record_filename):
                print('loading testing records from', record_filename)
                record_dict = joblib.load(record_filename)
            else:
                print('creating testing records...')
                record_dict = {}
            for config_idx, config in enumerate(route_configurations):
                print('config {}/{}'.format(config_idx + 1, len(route_configurations)))
                if config.data_id in record_dict and len(record_dict[config.data_id]) > 0:
                    print('skipping tested scenario', config.data_id)
                    continue
                for repeat in range(self._args.repetitions):
                    result, record = self._load_and_run_scenario(config)
                    if result:
                        record_dict[config.data_id] = record
                        print('saving testing records to', record_filename)
                        joblib.dump(record_dict, record_filename)
                        print(config.data_id, 'saved!', 'length:', len(record))
                    self._cleanup()
        self._cleanup()
        return result

    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        os.makedirs(self._args.outputDir, exist_ok=True)
        result = self._run_route()
        print("No more scenarios .... Exiting")
        return result


def main():
    """
    main function
    """
    description = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + VERSION)

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default="10.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')
    parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')

    parser.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    parser.add_argument(
        '--route', help='Run a route as a scenario (input: (route_file,scenario_file,[route id]))', nargs='+', type=str)

    parser.add_argument(
        '--agent', help="Agent used to execute the scenario. Currently only compatible with route-based scenarios.")
    parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument('--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--file', action="store_true", help='Write results into a txt file')
    parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
    parser.add_argument('--json', action="store_true", help='Write results into a JSON file')
    parser.add_argument('--outputDir', default='/home/carla/output', help='Directory for output files (default: this directory)')

    parser.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    parser.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')

    parser.add_argument('--debug', action="store_true", help='Run with debug output')
    parser.add_argument('--reloadWorld', action="store_true",
                        help='Reload the CARLA world before starting a scenario (default=True)')
    parser.add_argument('--record', type=str, default='',
                        help='Path were the files will be saved, relative to SCENARIO_RUNNER_ROOT.\nActivates the CARLA recording feature and saves to file all the criteria information.')
    parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    parser.add_argument('--repetitions', default=10, type=int, help='Number of scenario executions')
    parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    parser.add_argument('--train_agent', action="store_true", help='Train RL agents')
    parser.add_argument('--train_agent_episodes', default=100, type=int, help='Training episodes')

    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    set_seed(arguments.seed)


    rospy.init_node('scenario_runner', anonymous=True)
    get_ego_vehicle_route_service = rospy.Service('/carla_data_provider/get_ego_vehicle_route',
                                                  GetEgoVehicleRoute, CarlaDataProvider.get_ego_vehicle_route_callback)

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if not arguments.scenario and not arguments.openscenario and not arguments.route:
        print("Please specify either a scenario or use the route mode\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.route and (arguments.openscenario or arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.route:
        arguments.reloadWorld = False

    if arguments.agent:
        arguments.sync = True

    scenario_runner = None
    result = True
    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()

    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result


if __name__ == "__main__":
    sys.exit(main())
