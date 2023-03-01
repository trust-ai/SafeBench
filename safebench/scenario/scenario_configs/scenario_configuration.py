'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:43:48
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarioconfigs/scenario_configuration.py>
    Copyright (c) 2019 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla


class ActorConfigurationData(object):
    """
        This is a configuration base class to hold model and transform attributes
    """
    def __init__(self, model, transform, rolename='other', speed=0, autopilot=False, random=False, color=None, category="car", args=None):
        self.model = model
        self.rolename = rolename
        self.transform = transform
        self.speed = speed
        self.autopilot = autopilot
        self.random_location = random
        self.color = color
        self.category = category
        self.args = args

    @staticmethod
    def parse_from_node(node, rolename):
        model = node.attrib.get('model', 'vehicle.*')

        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))
        yaw = float(node.attrib.get('yaw', 0))

        transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw))
        rolename = node.attrib.get('rolename', rolename)
        speed = node.attrib.get('speed', 0)

        autopilot = False
        if 'autopilot' in node.keys():
            autopilot = True

        random_location = False
        if 'random_location' in node.keys():
            random_location = True

        color = node.attrib.get('color', None)

        return ActorConfigurationData(model, transform, rolename, speed, autopilot, random_location, color)


class ScenarioConfiguration(object):
    """
        This class provides a basic scenario configuration incl.:
            - configurations for all actors
            - town, where the scenario should be executed
            - name of the scenario (e.g. ControlLoss_1)
            - type is the class of scenario (e.g. ControlLoss)
    """

    num_scenario = None
    trigger_points = []
    ego_vehicles = []
    other_actors = []
    town = None
    name = None
    type = None
    route = None
    agent = None
    weather = carla.WeatherParameters()
    friction = None
    subtype = None
    route_var_name = None

    parameters = None
