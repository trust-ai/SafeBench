'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:42:56
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarioconfigs/route_scenario_configuration.py>
    Copyright (c) 2019 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from agents.navigation.local_planner import RoadOption
from safebench.scenario.scenario_configs.scenario_configuration import ScenarioConfiguration


class RouteConfiguration(object):
    """
        This class provides the basic  configuration for a route
    """

    def __init__(self, route=None):
        self.data = route

    def parse_xml(self, node):
        self.data = []

        for waypoint in node.iter("waypoint"):
            x = float(waypoint.attrib.get('x', 0))
            y = float(waypoint.attrib.get('y', 0))
            z = float(waypoint.attrib.get('z', 0))
            c = waypoint.attrib.get('connection', '')
            connection = RoadOption[c.split('.')[1]]
            self.data.append((carla.Location(x, y, z), connection))


class RouteScenarioConfiguration(ScenarioConfiguration):
    """
        Basic configuration of a RouteScenario
    """
    auto_ego = False
    num_scenario = None
    trajectory = None
    scenario_file = None
    scenario_config = None

    initial_transform = None
    initial_pose = None

    data_id = 0
    scenario_generation_method = None
    scenario_id = 0
    route_id = 0
    risk_level = 0
    parameters = None
