#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function

import os
import sys
import time
import matplotlib.pyplot as plt

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0):
        """
        Setups up the parameters, which will be filled at load_scenario()

        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        # TODO: Properly remove this line to destroy actors
        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        if self._agent is not None:
            self._sync_mode = True
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print(f">>>>>>>>>> run_scenario: Scenario status {self.scenario_tree.status}")
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        # ego_location_list = []
        # actor_location_list = []

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)
                print("scenario definations: ", self.scenario_class.sampled_scenarios_definitions)
                print("ego position: ", CarlaDataProvider.get_transform(self.ego_vehicles[0]))
                # actors = list(CarlaDataProvider.get_actors())
                # ego_location = actors[0][1].get_location()
                # ego_location_list.append([ego_location.x, ego_location.y, ego_location.z])
                # actor_location = actors[1][1].get_location()
                # actor_location_list.append([actor_location.x, actor_location.y, actor_location.z])
                # if len(ego_location_list) % 100 == 0:
                #     fig = plt.figure()
                #     ax = fig.gca()
                #     ax.set_title("Route")
                #     ax.set_xlabel("x")
                #     ax.set_ylabel("y")
                #     ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()
                #     ax.plot([item[0].x for item in ego_vehicle_route], [item[0].y for item in ego_vehicle_route], c='g')
                #     ax.plot([item[0] for item in ego_location_list], [item[1] for item in ego_location_list], c='r')
                #     # plt.scatter(x=[item[0] for item in ego_location_list], y=[item[1] for item in ego_location_list], marker='x', color='red')
                #     ax.plot([item[0] for item in actor_location_list], [item[1] for item in actor_location_list], c='b')
                #     # plt.scatter(x=[item[0] for item in actor_location_list], y=[item[1] for item in actor_location_list], marker='x', color='blue')
                #     # plt.scatter(x=[item.x for item in config.trajectory], y=[item.y for item in config.trajectory], marker='x', color='red')
                #     # # plt.scatter(x=[p[0] for p in t], y=[p[1] for p in t], marker='x', color='blue')
                #     # plt.scatter(x=[p[0] for p in mt], y=[p[1] for p in mt], marker='x', color='blue')
                #     os.makedirs('/home/carla/Evaluation/locations', exist_ok=True)
                #     plt.savefig('/home/carla/Evaluation/locations/locations_%04d.png' % len(ego_location_list))
                #     plt.close()
                #     # if len(ego_location_list) % 500 == 0:
                #     #     print(ego_location_list)

        self._watchdog.stop()

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """
        # print('ego vehicle location: {}'.format(list(CarlaDataProvider._actor_transform_map.keys())[0].get_location()))
        # raise Exception
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            # print(f">>>>>>>>>> tick_scenario: Scenario status {self.scenario_tree.status}")
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # if self._agent is not None:
                # ego_action = self._agent()

            # if self._agent is not None:
            #     self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit, json):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit, json)
        output.write()

        return failure or timeout
