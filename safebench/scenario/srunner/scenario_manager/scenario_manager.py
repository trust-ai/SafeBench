'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-14 12:07:32
Description: 
'''

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenario_manager.timer import GameTime
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_locations


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.

        The user must not modify this class.
            To use the ScenarioManager:
            1. Create an object via manager = ScenarioManager()
            2. Load a scenario via manager.load_scenario()
            4. Trigger a result evaluation with manager.analyze_scenario()
            5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, logger):
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None
        self.scenario_list = None
        self.triggered_scenario = None
        self.logger = logger
        self._reset()

    def _reset(self):
        self._running = False
        self._timestamp_last_run = 0.0
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
        self._running = True
        self._init_scenarios()

    def _init_scenarios(self):
        # spawn background actors
        self.scenario_class.initialize_actors()
        # spawn actors for each scenario
        for i in range(len(self.scenario_list)):
            self.scenario_list[i].initialize_actors()
            self.scenario_class.other_actors += self.scenario_list[i].other_actors

    def stop_scenario(self):
        self._running = False

    def update_running_status(self):
        record, stop = self.scenario_class.get_running_status(self.running_record)
        self.running_record.append(record)
        if stop:
            self._running = False

    def get_update(self, timestamp):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
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
                        self.logger.log(">> Trigger scenario: " + spawned_scenario.name)
                        self.triggered_scenario.add(spawned_scenario)

            for running_scenario in self.triggered_scenario.copy(): # why using copy?
                #TODO: update behavior of agents in scenario
                running_scenario.update_behavior()

            self.update_running_status()

    def evaluate(self, ego_action, world_2_camera, image_w, image_h, fov, obs):
        # try:
        bbox_pred = ego_action['od_result']
        self.scenario_class.get_bbox(world_2_camera, image_w, image_h, fov)
        bbox_label = self.scenario_class.ground_truth_bbox
        self.scenario_class.eval(bbox_pred, bbox_label)
        self.scenario_class.save_img_label(obs, bbox_label)
        # print(bbox_pred, bbox_label)
        print('evaluate finished') # TODO
        # except:
        #     print('evaluate errors!')
        #     pass