#import py_trees

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import TimeOut
from safebench.scenario.scenario_manager.weather_sim import WeatherBehavior


class SpawnOtherActorError(Exception):
    pass


class BasicScenario(object):
    """
        Base class for user-defined scenario
    """
    def __init__(self, name, config, world, first_env=False):
        self.world = world
        self.other_actors = []
        self.actor_type_list = []
        self.other_actor_transform = []

        self.trigger_distance_threshold = None
        self.reference_actor = None

        self.name = name
        self.config = config

        if first_env:
            self._initialize_environment(world)

        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        '''
        behavior = self.create_behavior()
        behavior_seq = py_trees.composites.Sequence()
        if behavior is not None:
            behavior_seq.add_child(behavior)
            behavior_seq.name = behavior.name
        self.scenario = ScenarioDynamic(behavior_seq, self.name)
        '''

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

    def create_behavior(self):
        """
            This method just for background defination in route scenaio
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def clean_up(self):
        """
            Remove all actors
        """
        # TODO: destroy collision sensors before destroying actors
        for s_i in range(len(self.other_actors)):
            if self.other_actors[s_i].type_id.startswith('vehicle'):
                self.other_actors[s_i].set_autopilot(enabled=False)
            if CarlaDataProvider.actor_id_exists(self.other_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.other_actors[s_i].id)
        self.other_actors = []


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

    def __init__(self, behavior, name, timeout=60):
        self.behavior = behavior
        self.timeout = timeout
        self.name = name

        self.test_criteria = None

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        # TODO: remove py_tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SuccessOnOne)
        if behavior is not None:
            self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)
        self.scenario_tree.add_child(WeatherBehavior())
        # self.scenario_tree.add_child(UpdateAllActorControls())

        self.scenario_tree.setup(timeout=1)
