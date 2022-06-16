import os
import os.path as osp
from copy import deepcopy
from fnmatch import fnmatch

import numpy as np
import yaml


def print_dict(d):
    print(yaml.dump(d, sort_keys=False, default_flow_style=False))


def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


'''
********************** eval utils ************************
'''


def find_config_dir(dir, depth=0):
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name == "config.yaml":
                return path, name
    # if we can not find the config file from the current dir, we search for the parent dir:
    if depth > 2:
        raise ValueError(
            "We can not find 'config.yaml' from your provided dir and its parent dirs!")
    return find_config_dir(osp.dirname(dir), depth + 1)


def find_model_path(dir, itr=None):
    # if itr is specified, return model with the itr number
    if itr is not None:
        model_path = osp.join(dir, "model_" + str(itr) + ".pt")
        if not osp.exists(model_path):
            raise ValueError("Model doesn't exist: " + model_path)
        return model_path
    # if itr is not specified, return model.pt or the one with the largest itr number
    pattern = "*pt"
    model = "model.pt"
    max_itr = -1
    for _, _, files in os.walk(dir):
        for name in files:
            if fnmatch(name, pattern):
                name = name.split(".pt")[0].split("_")
                if len(name) > 1:
                    itr = int(name[1])
                    if itr > max_itr:
                        max_itr = itr
                        model = "model_" + str(itr) + ".pt"
    model_path = osp.join(dir, model)
    if not osp.exists(model_path):
        raise ValueError("Model doesn't exist: " + model_path)
    return model_path


def setup_eval_configs(dir, itr=None):
    '''
    Given an experiment dir, return the model path and experiment configs.
    It will automatically search the folder and its parent folder to locate the first 
    `config.yaml` that it can found
    @param dir (str), something like '../data/SafetyCarCircle_sac'
    @param itr (int), the model iteration number. If None, return the one with maximum itr

    @return model path, env name, policy name, timeout_steps info
    '''
    path, config_name = find_config_dir(dir)
    model_path = find_model_path(osp.join(path, "model_save"), itr=itr)
    config_path = osp.join(path, config_name)
    print("Model path: %s " % model_path)
    print("Loading config from %s " % config_path)
    configs = load_config(config_path)
    return model_path, configs["policy"], configs["timeout_steps"], configs[
        configs["policy"]]
