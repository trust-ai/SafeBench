''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-05 14:14:23
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import os.path as osp
import time
from fnmatch import fnmatch

import yaml
import imageio


def save_gif(frame_list, filename):
    imageio.v2.mimsave(filename, frame_list, fps=30)


class VideoRecorder(object):
    def __init__(self, output_dir, logger):
        self.logger = logger
        self.output_dir = output_dir
        self.video_count = 0
        self.frame_list = []
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.video_dir = os.path.join(self.output_dir, 'video', hms_time)

    def add_frame(self, frame):
        self.frame_list.append(frame)
    
    def save(self, data_ids):
        data_ids = ['{:04d}'.format(data_id) for data_id in data_ids]
        video_name = f'video_{"{:04d}".format(self.video_count)}_id_{"_".join(data_ids)}.gif'
        os.makedirs(self.video_dir, exist_ok=True)
        video_file = os.path.join(self.video_dir, video_name)
        self.logger.log(f'>> Saving video to {video_file}')
        save_gif(self.frame_list, video_file)
        self.frame_list = []
        self.video_count += 1


def print_dict(d):
    print(yaml.dump(d, sort_keys=False, default_flow_style=False))


def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_config_dir(dir, depth=0):
    for path, subdirs, files in os.walk(dir):
        for name in files:
            if name == "config.yaml":
                return path, name
    # if we can not find the config file from the current dir, we search for the parent dir:
    if depth > 2:
        return None
    return find_config_dir(osp.dirname(dir), depth + 1)


def find_model_path(dir, itr=None):
    # if itr is specified, return model with the itr number
    if itr is not None:
        model_path = osp.join(dir, "model_" + str(itr) + ".pt")
        if not osp.exists(model_path):
            return None
            # raise ValueError("Model doesn't exist: " + model_path)
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
        return None
        # raise ValueError("Model doesn't exist: " + model_path)
    return model_path, max_itr


def setup_eval_configs(dir, itr=None):
    path, config_name = find_config_dir(dir)
    model_path, load_itr = find_model_path(osp.join(path, "model_save"), itr=itr)
    config_path = osp.join(path, config_name)
    configs = load_config(config_path)
    return model_path, load_itr, configs["policy"], configs["timeout_steps"], configs[configs["policy"]]
