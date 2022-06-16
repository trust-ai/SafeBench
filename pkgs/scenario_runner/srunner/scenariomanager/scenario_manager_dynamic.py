"""
@author: Shuai Wang
@e-mail: ws199807@outlook.com
This module provides the dynamic version ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time
import carla

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.AdditionTools.scenario_utils import calculate_distance_transforms
from srunner.AdditionTools.scenario_utils import calculate_distance_locations

# # distance threshold for trigerring small scenarios in route
# distance_threshold = 15


class ScenarioManagerDynamic(object):

    """
    Dynamic version scenario manager class. This class holds all functionality
    required to initialize, trigger, update and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control init, trigger, update and stop
       of the scenario
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
        self._watchdog = None
        self._timeout = timeout

        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

        self.scenario_list = None
        self.triggered_scenario = None

        self.running_record = []

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
        self.running_record = []
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

        # if self.scenario is not None:
        #     self.scenario.terminate()

        if self.scenario_class is not None:
            self.scenario_class.__del__()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

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

        # all spawned scenarios on route
        self.scenario_list = scenario.list_scenarios
        # triggered scenario set
        self.triggered_scenario = set()

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog = Watchdog(float(self._timeout))
        self._watchdog.start()
        self._running = True

        # ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()
        # ego_vehicle_route = [item[0] for item in ego_vehicle_route]
        # for i in ego_vehicle_route:
        #     print(i)

        """First spawn all actors on the map"""
        self._init_scenarios()

        # actors = list(CarlaDataProvider.get_actors())
        # print('ego', actors[0][1].get_speed_limit())
        # if len(actors) > 1:
        #     print('other', actors[1][1].get_speed_limit())
        # if len(actors) > 2:
        #     print('other 2', actors[2][1].get_speed_limit())

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            # print("scenario defination: ", self.scenario_class.sampled_scenarios_definitions)
            # print("ego transform: ", CarlaDataProvider.get_transform(self.ego_vehicles[0]))
            if timestamp:
                self._get_update(timestamp)

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

        return self.running_record

    def _init_scenarios(self):
        # spawn background actors
        self.scenario_class.initialize_actors()
        # spawn actors for each scenario
        if self._running:
            for i in range(len(self.scenario_list)):
                self.scenario_list[i].initialize_actors()

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status() if self._watchdog is not None else False

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

    def update_running_status(self):
        record, stop = self.scenario_class.get_running_status(self.running_record)
        self.running_record.append(record)
        if stop:
            self._running = False

    def _get_update(self, timestamp):
        """
        This function used to trigger, update and stop scenario

        """
        """
        testing erea
        """

        """
        """
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # Tick scenario
            # note: for the background
            self.scenario_tree.tick_once()

            for spawned_scenario in self.scenario_list:
                ego_location = CarlaDataProvider.get_location(self.ego_vehicles[0])
                cur_distance = None
                reference_location = None
                if spawned_scenario.reference_actor:
                    reference_location = CarlaDataProvider.get_location(spawned_scenario.reference_actor)
                if reference_location:
                    cur_distance = calculate_distance_locations(ego_location, reference_location)

                if cur_distance and cur_distance < spawned_scenario.trigger_distance_threshold:
                    if spawned_scenario not in self.triggered_scenario:
                        print("Trigger scenario: " + spawned_scenario.name)
                        self.triggered_scenario.add(spawned_scenario)

            for running_scenario in self.triggered_scenario.copy():
                """
                update behavior
                """
                # print("Running scenario: " + running_scenario.name)
                running_scenario.update_behavior()
                # pass
                # comment: stop the scenario in certain condition, update triggered_scenario into not triggered
                # if running_scenario.check_stop_condition():
                #     print("Scenario" + running_scenario.name + "should stop")
                #     running_scenario.remove_all_actors()
                #     self.triggered_scenario.remove(running_scenario)
                #     self._running = False

            self.update_running_status()

            # actors = list(CarlaDataProvider.get_actors())
            # print('ego', actors[0][1].get_transform())
            # # print('ego', actors[0][1].get_velocity())
            # if len(actors) > 1:
            #     print('other', actors[1][1].get_transform())
            #     # print('other', actors[1][1].get_velocity())
            # if len(actors) > 2:
            #     print('other 2', actors[2][1].get_transform())
            #     # print('other 2', actors[2][1].get_velocity())

        if self._sync_mode and self._running and self._watchdog.get_status():
            # tick other parts in scenarios
            CarlaDataProvider.get_world().tick()
