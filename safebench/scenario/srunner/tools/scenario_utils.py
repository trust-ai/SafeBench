'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2023-01-30 22:30:39
LastEditTime: 2023-02-04 17:57:46
Description: 
'''
import math
import os.path as osp
import json
from .route_parser import RouteParser


def calculate_distance_transforms(transform_1, transform_2):
    distance_x = (transform_1.location.x - transform_2.location.x) ** 2
    distance_y = (transform_1.location.y - transform_2.location.y) ** 2
    return math.sqrt(distance_x + distance_y)


def calculate_distance_locations(location_1, location_2):
    distance_x = (location_1.x - location_2.x) ** 2
    distance_y = (location_1.y - location_2.y) ** 2
    return math.sqrt(distance_x + distance_y)


def scenario_parse(ROOT_DIR, config):
    """
    data file should also come from args
    """
    print("######## parsing scenario route and data ########")

    data_file = osp.join(ROOT_DIR, config['data_path'], config['method']+'.json')
    print('Using data file:', data_file)
    route_file_formatter = ROOT_DIR + '/' + config['route_path'] + '/scenario_%02d_routes/scenario_%02d_route_%02d.xml'
    scenario_file_formatter = ROOT_DIR + '/' + config['route_path'] + '/scenarios/scenario_%02d.json'
    
    # scenario_id, method, route_id, risk_level
    with open(data_file, 'r') as f:
        data_full = json.loads(f.read())
        data_full = [item for item in data_full if item["scenario_id"] == config['scenario_id']]
        data_full = [item for item in data_full if item["method"] == config['method']]

    print('loading {} data'.format(len(data_full)))
    map_town_config = {}
    route_configurations = []
    for item in data_full:
        route_file = route_file_formatter % (item['scenario_id'], item['scenario_id'], item['route_id'])
        scenario_file = scenario_file_formatter % item['scenario_id']
        parsed_configs = RouteParser.parse_routes_file(route_file, scenario_file)
        assert len(parsed_configs) == 1, item
        config = parsed_configs[0]
        config.data_id = item['data_id']
        config.scenario_generation_method = item['method']
        config.scenario_id = item['scenario_id']
        config.route_id = item['route_id']
        config.risk_level = item['risk_level']
        config.parameters = item['parameters']
        route_configurations.append(config)

        # build town and config mapping map
        cur_town = config.town
        if cur_town in map_town_config:
            cur_config_list = map_town_config[cur_town]
            cur_config_list.append(config)
            map_town_config[cur_town] = cur_config_list
        else:
            cur_config_list = [config]
            map_town_config[cur_town] = cur_config_list

    return route_configurations, map_town_config

