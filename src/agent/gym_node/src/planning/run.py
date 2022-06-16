#!/usr/bin/env
import rospy
import os.path as osp

from planning.carla_runner import CarlaRunner
from planning.safe_rl.util.run_util import load_config

from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus

scenario_status = CarlaScenarioStatus.STOPPED


CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")
RECORD_DIR = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(
        osp.dirname(osp.dirname(osp.dirname(
            osp.realpath(__file__)))))))), "output/videos")

EXP_NAME_KEYS = {"epochs": "epoch", "obs_type": "obs_type"}
DATA_DIR_KEYS = {"cost_limit": "cost"}
RECORD_DIR_KEYS = {"obs_type": "obs-type"}

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

def gen_record_dir_name(onfig: dict):
    name = config["policy"]
    for k in RECORD_DIR_KEYS:
        name += '_' + RECORD_DIR_KEYS[k] + '_' + str(config[k])
    return osp.join(RECORD_DIR, name)

if __name__ == '__main__':
    rospy.init_node('gym_node', anonymous=True)
    args_dict = dict()
    args_dict["policy"] = rospy.get_param("~policy", "sac")
    args_dict["load_dir"] = rospy.get_param("~load_dir", None)
    args_dict["mode"] = rospy.get_param("~mode", "eval")
    args_dict["seed"] = int(rospy.get_param("~seed", 0))
    args_dict["device"] = rospy.get_param("~device", "gpu")
    args_dict["epochs"] = int(rospy.get_param("~epochs", 4000))
    args_dict["suffix"] = None
    args_dict["port"] = int(rospy.get_param('~port', 2000))
    args_dict["sample_episode_num"] = int(rospy.get_param("~sample_episode_num", 10))
    args_dict["role_name"] = rospy.get_param("~role_name", "ego_vehicle")
    args_dict["continue_from_epoch"] = int(rospy.get_param("~continue_from_epoch", 0))
    args_dict["obs_type"] = int(rospy.get_param("~obs_type", 0))

    cofig_file = rospy.get_param("~config_file", "config_carla.yaml")
    config_path = osp.join(CONFIG_DIR, cofig_file)
    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args_dict["suffix"])
    config["data_dir"] = gen_data_dir_name(config)
    config["record_dir"] = gen_record_dir_name(config)

    runner = CarlaRunner(**config)

    # runner = SafetyGymRunner(**config)
    if args_dict["mode"] == "train":
        runner.train()
    else:
        runner.eval(render=False, sleep=0)
