import os.path as osp

from planning.carla_runner import CarlaRunner
from planning.safe_rl.util.run_util import load_config

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")

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

    args = parser.parse_args()
    args_dict = vars(args)

    config_path = osp.join(CONFIG_DIR, "config_carla.yaml")
    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    config["port"] = args.port
    config["traffic_port"] = args.traffic_port

    runner = CarlaRunner(**config)

    # runner = SafetyGymRunner(**config)
    if args.mode == "train":
        runner.train()
    else:
        runner.eval(render=False, sleep=0)