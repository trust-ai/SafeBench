from __future__ import print_function

import math
import carla
import random

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.tools.scenario_helper import get_location_in_distance_from_wp


class NormalTrainingScenario(BasicScenario):

    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()
        self.trigger_distance_threshold = 100
        self.number_of_vehicles = int(100 / config.num_scenario)
        self.number_of_walkers = 0

        super(NormalTrainingScenario, self).__init__("NormalTrainingScenario",
                                                    ego_vehicles,
                                                    config,
                                                    world)
        self.reference_actor = self.ego_vehicles[0]

    def initialize_actors(self):

        """
        Set other_actors to the superset of all scenario actors
        """
        actors = CarlaDataProvider.request_new_batch_actors('vehicle.*', amount=self.number_of_vehicles,
                                                            spawn_points=None, autopilot=True,
                                                            random_location=True, rolename='autopilot')
        self.other_actors += actors

        # walker_spawn_points = []
        # for i in range(self.number_of_walkers):
        #     spawn_point = carla.Transform()
        #     loc = self.world.get_random_location_from_navigation()
        #     if (loc != None):
        #         spawn_point.location = loc
        #         walker_spawn_points.append(spawn_point)

        # Spawn surrounding vehicles
        # vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # random.shuffle(vehicle_spawn_points)
        # count = self.number_of_vehicles
        # if count > 0:
        #     for spawn_point in vehicle_spawn_points:
        #         vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
        #         if vehicle is not None:
        #             count -= 1
        #             self.other_actors.append(vehicle)
        #         if count <= 0:
        #             break
        # while count > 0:
        #     vehicle = self._try_spawn_random_vehicle_at(random.choice(vehicle_spawn_points), number_of_wheels=[4])
        #     if vehicle is not None:
        #         count -= 1
        #         self.other_actors.append(vehicle)

        # Spawn pedestrians
        # random.shuffle(walker_spawn_points)
        # count = self.number_of_walkers
        # if count > 0:
        #     for spawn_point in walker_spawn_points:
        #         walker = self._try_spawn_random_walker_at(spawn_point)
        #         if walker is not None:
        #             count -= 1
        #             self.other_actors.append(walker)
        #         if count <= 0:
        #             break
        # while count > 0:
        #     walker = self._try_spawn_random_walker_at(random.choice(walker_spawn_points))
        #     if walker is not None:
        #         count -= 1
        #         self.other_actors.append(walker)


    def update_behavior(self, scenario_action):
        pass

    def check_stop_condition(self):
        """
        Now use distance actor[0] runs
        """
        return False

    def create_behavior(self, scenario_init_action):
        pass

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
        return walker_actor

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
        return vehicle

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.
        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp