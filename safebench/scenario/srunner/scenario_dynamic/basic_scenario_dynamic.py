"""
@author: Shuai Wang from SafeAI lab in CMU
@e-mail: ws199807@outloook.com

This module provides BasicScenarioDynamic, basic class for all interactive/dynamic scenarios
"""

from __future__ import print_function

import operator
import py_trees

import carla

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenario_manager.timer import TimeOut
from safebench.scenario.srunner.scenario_manager.weather_sim import WeatherBehavior
# from scenario_runner.srunner.scenario_manager.scenarioatomics.atomic_behaviors import UpdateAllActorControls

class SpawnOtherActorError(Exception):
    pass

class BasicScenarioDynamic(object):
    """
    Base class for user-defined scenario
    """
    def __init__(self, name, ego_vehicles, config, world, debug_mode=False, terminate_on_failure=False, criteria_enable=False, first_env=False):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.world = world
        self.other_actors = []
        self.actor_type_list = []
        self.other_actor_transform = []
        """for triggering"""
        self.trigger_distance_threshold = None
        self.reference_actor = None

        if not self.timeout:     # pylint: disable=access-member-before-definition
            self.timeout = 60    # If no timeout was provided, set it to 60 seconds

        self.scenario = None

        self.ego_vehicles = ego_vehicles
        self.name = name
        self.config = config
        self.terminate_on_failure = terminate_on_failure
        if first_env:
            self._initialize_environment(world)

        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # Setup scenario
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        behavior = self._create_behavior()

        behavior_seq = py_trees.composites.Sequence()

        if behavior is not None:
            behavior_seq.add_child(behavior)
            behavior_seq.name = behavior.name

        self.scenario = ScenarioDynamic(behavior_seq, self.name, self.timeout, self.terminate_on_failure)

    def _initialize_environment(self, world):
        """
        Default initialization of weather and road friction.
        Override this method in child class to provide custom initialization.
        """

        # Set the appropriate weather conditions
        world.set_weather(self.config.weather)

        # Set the appropriate road friction
        if self.config.friction is not None:
            friction_bp = world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(self.config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            world.spawn_actor(friction_bp, transform)

    def _create_behavior(self):
        """
        This method just for background defination in route scenaio
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios"
                "If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios"
                "If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if self.other_actors[i].type_id.startswith('vehicle'):
                    self.other_actors[i].set_autopilot(enabled=False)
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []

    def _del(self):
        self.remove_all_actors()


class ScenarioDynamic(object):
    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The scenario_tree is for the whole world, not for the specific actors in scenarios

    Maintaining scenario_tree is for background ticking

    The user must not modify this class.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(self, behavior, name, timeout=60, terminate_on_failure=False):
        self.behavior = behavior
        self.timeout = timeout
        self.name = name

        self.test_criteria = None

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        """This scenario_tree is for whole world"""
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SuccessOnOne)
        if behavior is not None:
            self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)
        self.scenario_tree.add_child(WeatherBehavior())
        # self.scenario_tree.add_child(UpdateAllActorControls())

        self.scenario_tree.setup(timeout=1)
