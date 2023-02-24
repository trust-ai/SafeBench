import carla

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario
from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation


class ManeuverOppositeDirection(BasicScenario):
    """
        Vehicle Maneuvering In Opposite Direction (Traffic Scenario 06).
        Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions, 
        at a non-junction and encroaches into another vehicle traveling in the opposite direction.
    """

    def __init__(self, world, ego_vehicles, config, obstacle_type='vehicle', timeout=120):
        super(ManeuverOppositeDirection, self).__init__("ManeuverOppositeDirection", ego_vehicles, config, world)

        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 50
        self._second_vehicle_location = self._first_vehicle_location + 30
        self._opposite_speed = 8   # m/s
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        # obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        self._obstacle_type = obstacle_type
        self._first_actor_transform = None
        self._second_actor_transform = None
        self.timeout = timeout

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)

        self.actor_type_list.append('vehicle.nissan.micra')
        self.actor_type_list.append('vehicle.nissan.micra')

        self.reference_actor = None
        self.trigger_distance_threshold = 45
        self.ego_max_driven_distance = 200

    def initialize_actors(self):
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(first_actor_waypoint.transform.location, first_actor_waypoint.transform.rotation)
        self.other_actor_transform.append(first_actor_transform)
        self.other_actor_transform.append(second_actor_waypoint.transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]

    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        self.scenario_operation.go_straight(self._opposite_speed, 1)

    def check_stop_condition(self):
        pass
