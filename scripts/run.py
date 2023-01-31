import os.path as osp

import json
import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import seed_torch, export_device_env_variable
from safebench.carla_runner import CarlaRunner
from safebench.scenario.srunner.tools.route_parser import RouteParser

ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))


# TODO: move to util folder
def scenario_parse(config):
    """
    data file should also come from args
    """
    data_file = osp.join(ROOT_DIR, config['data_path'])
    if config['method'] == 'benign':
        data_file += '/benign.json'
    elif config['method'] == 'standard':
        data_file += '/standard.json'
    else:
        data_file += '/dev.json'

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

    print("######## Route parsing done ########")
    return route_configurations, map_town_config


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--num_scenario', type=int, default=1)

    args = parser.parse_args()
    args_dict = vars(args)

    # set some device parameters
    export_device_env_variable(args.device, id=args.device_id)
    torch.set_num_threads(args.threads)
    seed_torch(args.seed)

    # load agent config
    agent_config_path = osp.join(ROOT_DIR, "safebench/agent/config/example.yaml")
    agent_config = load_config(agent_config_path)

    # load scenario config
    scenario_config_path = osp.join(ROOT_DIR, "safebench/scenario/config/example.yaml")
    scenario_config = load_config(scenario_config_path)
    route_configurations, map_town_config = scenario_parse(scenario_config)
    scenario_config.update(args_dict)
    scenario_config["map_town_config"] = map_town_config

    # main entry with a selected mode
    runner = CarlaRunner(agent_config, scenario_config)
    runner.run()
