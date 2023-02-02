"""
@author: Shuai Wang
@e-mail: ws199807@outlook.com
This module provides the dynamic version ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import time
import carla
import py_trees

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenario_manager.timer import GameTime
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_locations


class ScenarioManagerDynamic(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.

        The user must not modify this class.
            To use the ScenarioManager:
            1. Create an object via manager = ScenarioManager()
            2. Load a scenario via manager.load_scenario()
            3. Trigger the execution of the scenario manager.run_scenario() This function is designed to explicitly control init, trigger, update and stop of the scenario
            4. Trigger a result evaluation with manager.analyze_scenario()
            5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self):
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

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
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.running_record = []
        GameTime.restart()

    def cleanup(self):
        if self.scenario_class is not None:
            self.scenario_class.__del__()

    def load_scenario(self, scenario):
        self._reset()
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # all spawned scenarios on route
        self.scenario_list = scenario.list_scenarios
        # triggered scenario set
        self.triggered_scenario = set()

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()

        self._running = True
        self._init_scenarios()

    def _init_scenarios(self):
        # spawn background actors
        self.scenario_class.initialize_actors()
        # spawn actors for each scenario
        for i in range(len(self.scenario_list)):
            self.scenario_list[i].initialize_actors()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def update_running_status(self):
        record, stop = self.scenario_class.get_running_status(self.running_record)
        self.running_record.append(record)
        if stop:
            self._running = False

    def get_update(self):
        CarlaDataProvider.on_carla_tick()
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

        for running_scenario in self.triggered_scenario.copy(): # why using copy?
            #TODO: update behavior of agents in scenario
            running_scenario.update_behavior()

        self.update_running_status()
