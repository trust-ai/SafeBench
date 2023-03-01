'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:54:10
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/tools>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math
import os.path as osp
import json
import random
from .route_parser import RouteParser


def calculate_distance_transforms(transform_1, transform_2):
    distance_x = (transform_1.location.x - transform_2.location.x) ** 2
    distance_y = (transform_1.location.y - transform_2.location.y) ** 2
    return math.sqrt(distance_x + distance_y)


def calculate_distance_locations(location_1, location_2):
    distance_x = (location_1.x - location_2.x) ** 2
    distance_y = (location_1.y - location_2.y) ** 2
    return math.sqrt(distance_x + distance_y)


def scenario_parse(config, logger):
    """
        Data file should also come from args
    """

    ROOT_DIR = config['ROOT_DIR']
    logger.log(">> Parsing scenario route and data")
    list_of_scenario_config = osp.join(ROOT_DIR, config['type_dir'], config['type_name'])
    route_file_formatter = osp.join(ROOT_DIR, config['route_dir'], 'scenario_%02d_routes/scenario_%02d_route_%02d.xml')
    scenario_file_formatter = osp.join(ROOT_DIR, config['route_dir'], 'scenarios/scenario_%02d.json')
    
    # scenario_id, method, route_id, risk_level
    with open(list_of_scenario_config, 'r') as f:
        data_full = json.loads(f.read())
        # filter the list if any parameter is specified
        if config['method'] is not None:
            logger.log('>> Selecting method: ' + config['method'])
            data_full = [item for item in data_full if item["method"] == config['method']]
        if config['scenario_id'] is not None:
            logger.log('>> Selecting scenario_id: ' + str(config['scenario_id']))
            data_full = [item for item in data_full if item["scenario_id"] == config['scenario_id']]
        if config['route_id'] is not None:
            logger.log('>> Selecting route_id: ' + str(config['route_id']))
            data_full = [item for item in data_full if item["route_id"] == config['route_id']]

    logger.log(f'>> Loading {len(data_full)} data')
    map_town_config = {}
    for item in data_full:
        route_file = route_file_formatter % (item['scenario_id'], item['scenario_id'], item['route_id'])
        scenario_file = scenario_file_formatter % item['scenario_id']
        parsed_configs = RouteParser.parse_routes_file(route_file, scenario_file)
        assert len(parsed_configs) == 1, item
        parsed_config = parsed_configs[0]
        parsed_config.auto_ego = config['auto_ego']
        parsed_config.num_scenario = config['num_scenario']
        parsed_config.data_id = item['data_id']
        parsed_config.scenario_generation_method = item['method']
        parsed_config.scenario_id = item['scenario_id']
        parsed_config.route_id = item['route_id']
        parsed_config.risk_level = item['risk_level']
        parsed_config.parameters = item['parameters']

        # build town and config mapping map
        cur_town = parsed_config.town
        if cur_town in map_town_config:
            cur_config_list = map_town_config[cur_town]
            cur_config_list.append(parsed_config)
            map_town_config[cur_town] = cur_config_list
        else:
            cur_config_list = [parsed_config]
            map_town_config[cur_town] = cur_config_list

    return map_town_config


def get_valid_spawn_points(world):
    vehicle_spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(vehicle_spawn_points)
    actor_location_list = get_current_location_list(world)
    vehicle_spawn_points = filter_valid_spawn_points(vehicle_spawn_points, actor_location_list)
    return vehicle_spawn_points


def filter_valid_spawn_points(spawn_points, current_locations):
    dis_threshold = 8
    valid_spawn_points = []
    for spawn_point in spawn_points:
        valid = True
        for location in current_locations:
            if spawn_point.location.distance(location) < dis_threshold:
                valid = False
                break
        if valid:
            valid_spawn_points.append(spawn_point)
    return valid_spawn_points


def get_current_location_list(world):
    locations = []
    for actor in world.get_actors().filter('vehicle.*'):
        locations.append(actor.get_transform().location)
    return locations
