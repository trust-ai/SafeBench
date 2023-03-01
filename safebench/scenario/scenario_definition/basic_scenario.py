'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-02-28 17:56:13
Description: 
'''

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class BasicScenario(object):
    """
        Base class for user-defined scenario
    """
    def __init__(self, name, config, world, first_env=False):
        self.world = world
        self.name = name
        self.config = config

        self.ego_vehicles = None
        self.reference_actor = None # the actor used for calculating trigger distance
        self.other_actors = []
        self.other_actor_transform = []
        self.trigger_distance_threshold = None
        self.ego_max_driven_distance = 200

        if first_env:
            self._initialize_environment(world)

        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

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

    def create_behavior(self, scenario_init_action):
        """
            This method just for background defination in route scenaio
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def initialize_actors(self):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def update_behavior(self, scenario_action):
        raise NotImplementedError(
                "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def check_stop_condition(self):
        raise NotImplementedError(
            "This function is re-implemented by all scenarios. If this error becomes visible the class hierarchy is somehow broken")

    def clean_up(self):
        """
            Remove all actors
        """
        for s_i in range(len(self.other_actors)):
            if self.other_actors[s_i].type_id.startswith('vehicle'):
                self.other_actors[s_i].set_autopilot(enabled=False, tm_port=CarlaDataProvider.get_traffic_manager_port())
            if CarlaDataProvider.actor_id_exists(self.other_actors[s_i].id):
                CarlaDataProvider.remove_actor_by_id(self.other_actors[s_i].id)
        self.other_actors = []
