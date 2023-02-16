from __future__ import print_function

import math
import numpy as np

import carla

from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario, SpawnOtherActorError
from safebench.scenario.srunner.tools.scenario_helper import (generate_target_waypoint,
                                           generate_target_waypoint_in_route,
                                           get_crossing_point,
                                           get_junction_topology)

from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_locations

from safebench.scenario.srunner.scenarios.LC.reinforce_continuous import REINFORCE, constraint, normalize_routes


def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
    Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    # offset = {"orientation": 270, "position": 190, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.x = trigger_location.x + 20
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
    Gets the driving / parking lane that is most to the right of the waypoint
    as well as the number of lane changes done
    """
    lane_changes = 0

    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next

    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
    This function filters false negative Shoulder which are in reality Parking lanes.
    These are differentiated from the others because, similar to the driving lanes,
    they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True

    return False


class VehicleTurningRoute(BasicScenario):
    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn. This is the version used when the ego vehicle
    is following a given route. (Traffic Scenario 4)
    This is a single ego vehicle scenario
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self.agent = REINFORCE(config=config.parameters)
        self._wmap = CarlaDataProvider.get_map()
        self.timeout = timeout
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        target_speed = 0.4

        route = []
        for point in self._ego_route:
            route.append([point[0].x, point[0].y])
        route = np.array(route)
        index = np.linspace(1, len(route) - 1, 30).tolist()
        index = [int(i) for i in index]
        route_norm = normalize_routes(route[index])
        route_norm = np.concatenate((route_norm, [[target_speed]]), axis=0)
        route_norm = route_norm.astype('float32')

        actions = self.agent.deterministic_action(route_norm)
        self.actions = actions
        print([i.item() for i in actions])
        self.running_distance = 10

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        # define scenario actions with scanerio modle agent
        # self.actions = actions

        super(VehicleTurningRoute, self).__init__(
            "VehicleTurningRouteDynamic",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
            terminate_on_failure=True
        )

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)

        self.actor_type_list.append('vehicle.diamondback.century')

        self.reference_actor = None
        self.trigger_distance_threshold = 20
        self.ego_max_driven_distance = 180

    def convert_actions(self, actions, x_scale, y_scale, x_mean, y_mean):
        yaw_min = 0
        yaw_max = 360
        yaw_scale = (yaw_max - yaw_min) / 2
        yaw_mean = (yaw_max + yaw_min) / 2

        d_min = 10
        d_max = 50
        d_scale = (d_max - d_min) / 2
        dist_mean = (d_max + d_min) / 2

        x = constraint(actions[0], -1, 1) * x_scale + x_mean
        y = constraint(actions[1], -1, 1) * y_scale + y_mean
        yaw = constraint(actions[2], -1, 1) * yaw_scale + yaw_mean
        dist = constraint(actions[3], -1, 1) * d_scale + dist_mean

        return [x, y, yaw, dist]

    def initialize_actors(self):
        """
        Custom initialization
        """
        cross_location = get_crossing_point(self.ego_vehicles[0])
        cross_waypoint = CarlaDataProvider.get_map().get_waypoint(cross_location)
        entry_wps, exit_wps = get_junction_topology(cross_waypoint.get_junction())
        assert len(entry_wps) == len(exit_wps)
        x = y = 0
        max_x_scale = max_y_scale = 0
        for i in range(len(entry_wps)):
            x += entry_wps[i].transform.location.x + exit_wps[i].transform.location.x
            y += entry_wps[i].transform.location.y + exit_wps[i].transform.location.y
        x /= len(entry_wps) * 2
        y /= len(entry_wps) * 2
        for i in range(len(entry_wps)):
            max_x_scale = max(max_x_scale, abs(entry_wps[i].transform.location.x - x), abs(exit_wps[i].transform.location.x - x))
            max_y_scale = max(max_y_scale, abs(entry_wps[i].transform.location.y - y), abs(exit_wps[i].transform.location.y - y))
        max_x_scale *= 0.8
        max_y_scale *= 0.8
        center_transform = carla.Transform(carla.Location(x=x, y=y, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))
        x_mean = x
        y_mean = y

        x, y, yaw, self.trigger_distance_threshold = self.convert_actions(self.actions, max_x_scale, max_y_scale, x_mean, y_mean)
        # x, y, yaw, self.trigger_distance_threshold = self.convert_action(self.actions)

        _other_actor_transform = carla.Transform(carla.Location(x, y, 0), carla.Rotation(yaw=yaw))
        self.other_actor_transform.append(_other_actor_transform)
        try:
            self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        except:
            raise SpawnOtherActorError

        """Also need to specify reference actor"""
        self.reference_actor = self.other_actors[0]

    def update_behavior(self):
        for i in range(len(self.other_actors)):
            cur_actor_target_speed = 10
            self.scenario_operation.go_straight(cur_actor_target_speed, i)

    def check_stop_condition(self):
        """
        This condition is just for small scenarios
        """

        return False

    def _create_behavior(self):
        pass
