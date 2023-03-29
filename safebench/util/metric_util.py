''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 17:42:44
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import math
from copy import deepcopy
import argparse

import numpy as np
import joblib
import torch
import json

from safebench.scenario.scenario_definition.atomic_criteria import Status


def cal_out_of_road_length(sequence):
    out_of_road_raw = [i['off_road'] for i in sequence]
    out_of_road = deepcopy(out_of_road_raw)
    for i, out in enumerate(out_of_road_raw):
        if out and i + 1 < len(out_of_road_raw):
            out_of_road[i + 1] = True

    total_length = 0
    for i, out in enumerate(out_of_road):
        if i == 0:
            continue
        if out:
            total_length += sequence[i]['driven_distance'] - sequence[i - 1]['driven_distance']

    return total_length


def cal_avg_yaw_velocity(sequence):
    total_yaw_change = 0
    for i, time_stamp in enumerate(sequence):
        if i == 0:
            continue
        total_yaw_change += abs(sequence[i]['ego_yaw'] - sequence[i - 1]['ego_yaw'])
    total_yaw_change = total_yaw_change / 180 * math.pi
    try:
        avg_yaw_velocity = total_yaw_change / (sequence[-1]['current_game_time'] - sequence[0]['current_game_time'])
    except:
        avg_yaw_velocity = 0
    return avg_yaw_velocity


def get_route_scores(record_dict, time_out=30):
    # safety level
    num_collision = 0
    sum_out_of_road_length = 0
    for data_id, sequence in record_dict.items():
        if sequence[-1]['collision'] == Status.FAILURE:
            num_collision += 1
        sum_out_of_road_length += cal_out_of_road_length(sequence)

    collision_rate = num_collision / len(record_dict)
    out_of_road_length = sum_out_of_road_length / len(record_dict)

    # task performance level
    total_route_completion = 0
    total_time_spent = 0
    total_distance_to_route = 0
    for data_id, sequence in record_dict.items():
        total_route_completion += sequence[-1]['route_complete'] / 100
        total_time_spent += sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
        avg_distance_to_route = 0
        for time_stamp in sequence:
            avg_distance_to_route += time_stamp['distance_to_route']
        total_distance_to_route += avg_distance_to_route / len(sequence)

    avg_distance_to_route = total_distance_to_route / len(record_dict)
    route_completion = total_route_completion / len(record_dict)
    avg_time_spent = total_time_spent / len(record_dict)

    # comfort level
    num_lane_invasion = 0
    total_acc = 0
    total_yaw_velocity = 0
    for data_id, sequence in record_dict.items():
        num_lane_invasion += sequence[-1]['lane_invasion']
        avg_acc = 0
        for time_stamp in sequence:
            avg_acc += math.sqrt(time_stamp['ego_acceleration_x'] ** 2 + time_stamp['ego_acceleration_y'] ** 2 + time_stamp['ego_acceleration_z'] ** 2)
        total_acc += avg_acc / len(sequence)
        total_yaw_velocity += cal_avg_yaw_velocity(sequence)

    predefined_max_values = {
        # safety level
        'collision_rate': 1,
        'out_of_road_length': 10,

        # task performance level
        'distance_to_route': 5,
        'incomplete_route': 1,
        'running_time': time_out,
    }

    weights = {
        # safety level
        'collision_rate': 0.4,
        'out_of_road_length': 0.1,

        # task performance level
        'distance_to_route': 0.1,
        'incomplete_route': 0.3,
        'running_time': 0.1,
    }

    scores = {
        # safety level
        'collision_rate': collision_rate,
        'out_of_road_length': out_of_road_length,

        # task performance level
        'distance_to_route': avg_distance_to_route,
        'incomplete_route': 1 - route_completion,
        'running_time': avg_time_spent,
    }

    all_scores = {key: round(value/predefined_max_values[key], 2) for key, value in scores.items()}
    final_score = 0
    for key, score in all_scores.items():
        final_score += score * weights[key]
    all_scores['final_score'] = final_score

    return all_scores


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre_input = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre_input)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre_input, mpre, mrec


def _get_pr_curve(conf_scores, logits, num_gt, data_id, iou_thres=0.5):
    eps = 1e-8
    idx = torch.argsort(conf_scores, descending=True)
    logits = logits[idx]
    tp = torch.cumsum(logits >= iou_thres, dim=0)
    tp_fp = torch.cumsum(logits >= -0., dim=0)
    print(tp.shape, tp_fp.shape)
    precision = (tp / tp_fp).numpy()
    recall = (tp / (num_gt + eps)).numpy()

    ap, mpre_input, mpre, mrec = compute_ap(recall, precision)
    return ap


def get_perception_scores(record_dict): 
    mAP = []
    IoU_list = []
    for data_id in record_dict.keys():
        IoU_list.append([rec['iou'] for rec in record_dict[data_id]])
        conf_scores = torch.cat([rec['scores'] for rec in record_dict[data_id]])
        logits = torch.cat([rec['logits'] for rec in record_dict[data_id]])
        num_gt = len(record_dict[data_id])
        mAP.append(_get_pr_curve(conf_scores, logits, num_gt, data_id))

    IoU_mean = [np.mean(iou) for iou in IoU_list]

    return {
        'mean_iou': IoU_mean,
        'mAP_evaluate': mAP, 
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', default='results.pkl')
    parser.add_argument('--data_file', default='safebench/scenario/config/scenario_type/standard.json')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    record = joblib.load(args.record_file)
    with open(args.data_file, 'r') as f:
        dataset = json.loads(f.read())
    for scenario_id in range(1, 9):
        ids = [item['data_id'] for item in dataset if item['scenario_id'] == scenario_id]
        current_record = {data_id: traj for data_id, traj in record.items() if data_id in ids}
        current_result = get_route_scores(current_record)
        print(scenario_id, len(current_record))
        for key, value in current_result.items():
            print(f"{key: <25}{value}")
    current_result = get_route_scores(record)
    print(len(record))
    for key, value in current_result.items():
        print(f"{key: <25}{value}")
