import os.path as osp

import json
import yaml

from safebench.util.run_util import load_config
from safebench.carla_runner_2 import CarlaRunner2 as CarlaRunner2
from safebench.scenario.srunner.tools.route_parser import RouteParser

UPPER_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))

EXP_NAME_KEYS = {"epochs": "epoch", "obs_type": "obs_type"}
DATA_DIR_KEYS = {"cost_limit": "cost"}


def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name + suffix


def gen_data_dir_name(config: dict):
    name = "carla"
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name


def get_scenario_configs(scenario_id, method):
    """
    data file should also come from args
    """
    data_file = osp.join(UPPER_DIR, 'safebench/scenario/scenario_data/data/')
    if method == 'benign':
        data_file += 'benign.json'
    elif method == 'standard':
        data_file += 'standard.json'
    else:
        data_file += 'dev.json'

    print('Using data file:', data_file)
    route_configurations = []
    route_file_formatter = UPPER_DIR + '/safebench/scenario/scenario_data/route/scenario_%02d_routes/scenario_%02d_route_%02d.xml'
    scenario_file_formatter = UPPER_DIR + '/safebench/scenario/scenario_data/route/scenarios/scenario_%02d.json'

    """
    scenario_id, method, route_id, risk_level
    """
    with open(data_file, 'r') as f:
        data_full = json.loads(f.read())
        data_full = [item for item in data_full if item["scenario_id"] == scenario_id]
        data_full = [item for item in data_full if item["method"] == method]

    print('loading {} data'.format(len(data_full)))
    map_town_config = {}
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', '-p', type=str, default='ppo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)
    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    ### Added for continue training when training process is interrupted
    parser.add_argument('--continue_from_epoch', '-c', type=int, default=0)
    parser.add_argument('--obs_type', '-o', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--traffic_port', type=int, default=2000)

    parser.add_argument('--scenario_id', type=int, default=5)
    parser.add_argument('--method', type=str, default='standard')
    parser.add_argument('--scenario_num', type=int, default=3)

    args = parser.parse_args()
    args_dict = vars(args)

    # config_path = osp.join(CONFIG_DIR, "config_carla.yaml")
    config_path = osp.join(UPPER_DIR, "safebench/agent/config/config_carla.yaml")
    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    config["port"] = args.port
    config["traffic_port"] = args.traffic_port

    route_configurations, map_town_config = get_scenario_configs(scenario_id=args.scenario_id, method=args.method)
    print("##### Route parsing done #####")

    config["map_town_config"] = map_town_config
    runner = CarlaRunner2(**config)

    # TODO: three modes, train agent (fix scenario), train scenario (fix agent), evaluate (fix all things)
    if args.mode == "train":
        runner.train()
    else:
        # runner.eval(render=False, sleep=0)
        runner.run_eval(scenario_num=args.scenario_num, render=False, sleep=0)
