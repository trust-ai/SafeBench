import carla
import random
import numpy as np

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario


class Detection_Pedestrian(BasicScenario):
    """
        This scenario create stopsign textures in the current scenarios.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(Detection_Pedestrian, self).__init__("Detection_Pedestrian", config, world)
        self._map = CarlaDataProvider.get_map()
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.timeout = timeout
        self.object_ad=list(filter(lambda k: 'AD' in k, world.get_names_of_all_objects())),
    
    def initialize_actors(self):
        """
        Initialize some background autopilot vehicles.
        """
        # actors = CarlaDataProvider.request_new_batch_actors(
        #     'vehicle.*', 
        #     amount=self.number_of_vehicles,
        #     spawn_points=None, autopilot=True,
        #     random_location=True, 
        #     rolename='autopilot'
        # )
        # self.other_actors = actors

        # the trigger distance will always be 0, trigger at the beginning
        self.reference_actor = self.ego_vehicle 

    def create_behavior(self, scenario_init_action):
        inputs = np.array(scenario_init_action['image'].detach().cpu().numpy()*255, dtype=np.int)
        height = 1024
        texture = carla.TextureColor(height,height)
        # TODO: run in multi-processing?
        for x in range(height):
            for y in range(height):
                r = int(inputs[x,y,0])
                g = int(inputs[x,y,1])
                b = int(inputs[x,y,2])
                a = int(255)
                # texture.set(x,height -0-y - 1, carla.Color(r,g,b,a))
                texture.set(height-x-1, height-y-1, carla.Color(r,g,b,a))
                # texture.set(x, y, carla.Color(r,g,b,a))
        for o_name in self.object_ad:
            self.world.apply_color_texture_to_object(o_name, carla.MaterialParameter.Diffuse, texture)
        
    def update_behavior(self, scenario_action):
        pass

    def check_stop_condition(self):
        return False

    def _try_spawn_random_walker_at(self, transform):
        """
            Try to spawn a walker at specific transform with random bluprint.
            Args:
                transform: the carla transform object.
            Returns:
                walker_actor: Bool indicating whether the spawn is successful.
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
        """
            Try to spawn a surrounding vehicle at specific transform with random bluprint.
            Args:
                transform: the carla transform object.
            Returns:
                vehicle: Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
        return vehicle

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """
            Create the blueprint for a specific actor type.
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