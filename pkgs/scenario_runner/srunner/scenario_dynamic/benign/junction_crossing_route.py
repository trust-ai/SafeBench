from __future__ import print_function
import carla
from srunner.AdditionTools.scenario_operation import ScenarioOperation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenario_dynamic.basic_scenario_dynamic import BasicScenarioDynamic
from srunner.AdditionTools.scenario_utils import calculate_distance_transforms


class OppositeVehicleRunningRedLightDynamic(BasicScenarioDynamic):
    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10

        super(OppositeVehicleRunningRedLightDynamic, self).__init__("OppositeVehicleRunningRedLightDynamic",
                                                             ego_vehicles,
                                                             config,
                                                             world,
                                                             debug_mode,
                                                             criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 0
        self.trigger = False
        self._actor_distance = 110
        self.ego_max_driven_distance = 150


    def initialize_actors(self):
        """
        Custom initialization
        """
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        self.other_actors[0].set_autopilot()

    def update_behavior(self):
        pass

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]),
                                                     self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionLeftTurnDynamic(BasicScenarioDynamic):
    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario
    An actor has higher priority, ego needs to yield to
    Oncoming actor
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
            Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12.0
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        super(SignalizedJunctionLeftTurnDynamic, self).__init__("TurnLeftAtSignalizedJunctionDynamic",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 0
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        """
        initialize actor
        """
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        # self.actor_type_list.append("vehicle.diamondback.century")
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        self.other_actors[0].set_autopilot()

    def update_behavior(self):
        pass

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class SignalizedJunctionRightTurnDynamic(BasicScenarioDynamic):
    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario
    An actor has higher priority, ego needs to yield to
    Oncoming actor
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
            Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 12
        self.timeout = timeout
        # self._brake_value = 0.5
        # self._ego_distance = 110
        self._actor_distance = 100
        super(SignalizedJunctionRightTurnDynamic, self).__init__("TurnRightAtSignalizedJunctionDynamic",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 0
        self.trigger = False
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        """
        initialize actor
        """
        config = self.config
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        self.other_actors[0].set_autopilot()

    def update_behavior(self):
        pass

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]), self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False


class NoSignalJunctionCrossingRouteDynamic(BasicScenarioDynamic):
    """

    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # Timeout of scenario in seconds
        self.timeout = timeout

        self.actor_speed = 10

        super(NoSignalJunctionCrossingRouteDynamic, self).__init__("NoSignalJunctionCrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)
        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.reference_actor = None
        self.trigger_distance_threshold = 0
        self.trigger = False

        self._actor_distance = 110
        self.ego_max_driven_distance = 150

    def initialize_actors(self):
        config = self.config
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.actor_type_list.append("vehicle.audi.tt")
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors,
                                                          self.actor_type_list)
        self.reference_actor = self.other_actors[0]
        self.other_actors[0].set_autopilot()

    def update_behavior(self):
        pass

    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        """
        small scenario stops when actor runs a specific distance
        """
        cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.other_actors[0]),
                                                     self.other_actor_transform[0])
        if cur_distance >= self._actor_distance:
            return True
        return False