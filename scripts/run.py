import os.path as osp

import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import seed_torch, set_torch_variable_env
from safebench.carla_runner import CarlaRunner
from safebench.scenario.srunner.tools.scenario_utils import scenario_parse


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--frame_skip', '-fs', type=int, default=4, help='skip of frame in each step')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--num_scenario', type=int, default=1, help='how many scenarios we will run parallelly')
    parser.add_argument('--num_episode', type=int, default=1, help='how many times one scenario will be repeated')
    args = parser.parse_args()
    args_dict = vars(args)

    # get the root dir of the package
    ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))

    # set some device parameters
    set_torch_variable_env(args.device)
    torch.set_num_threads(args.threads)
    seed_torch(args.seed)

    # load agent config
    agent_config_path = osp.join(ROOT_DIR, "safebench/agent/config/example.yaml")
    agent_config = load_config(agent_config_path)

    # load scenario config
    scenario_config_path = osp.join(ROOT_DIR, "safebench/scenario/config/example.yaml")
    scenario_config = load_config(scenario_config_path)
    route_configurations, map_town_config = scenario_parse(ROOT_DIR, scenario_config)
    scenario_config.update(args_dict)
    scenario_config["map_town_config"] = map_town_config

    # main entry with a selected mode
    runner = CarlaRunner(agent_config, scenario_config)
    runner.run()
