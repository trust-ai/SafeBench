import joblib
import math
from copy import deepcopy
import argparse
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Status


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
    avg_yaw_velocity = total_yaw_change / (sequence[-1]['current_game_time'] - sequence[0]['current_game_time'])

    return avg_yaw_velocity


def get_scores(record_dict):
    # safety level
    num_collision = 0
    num_run_red_light = 0
    num_run_stop_sign = 0
    sum_out_of_road_length = 0
    sum_route_length = 0
    for data_id, sequence in record_dict.items():
        if sequence[-1]['collision'] == Status.FAILURE:
            num_collision += 1
        num_run_red_light += sequence[-1]['run_red_light']
        num_run_stop_sign += sequence[-1]['run_stop']
        sum_out_of_road_length += cal_out_of_road_length(sequence)
        sum_route_length += sequence[-1]['driven_distance'] / sequence[-1]['route_complete'] * 100

    collision_rate = num_collision / len(record_dict)
    avg_red_light_freq = num_run_red_light / len(record_dict)
    avg_stop_sign_freq = num_run_stop_sign / len(record_dict)
    out_of_road_length = sum_out_of_road_length / len(record_dict)
    avg_route_length = sum_route_length / len(record_dict)

    # task performance level
    total_route_completion = 0
    total_time_spent = 0
    success_data_cnt = 0
    total_distance_to_route = 0
    for data_id, sequence in record_dict.items():
        total_route_completion += sequence[-1]['route_complete'] / 100
        if sequence[-1]['route_complete'] == 100:
            success_data_cnt += 1
            total_time_spent += sequence[-1]['current_game_time'] - sequence[0]['current_game_time']
        avg_distance_to_route = 0
        for time_stamp in sequence:
            avg_distance_to_route += time_stamp['distance_to_route']
        total_distance_to_route += avg_distance_to_route / len(sequence)

    avg_distance_to_route = total_distance_to_route / len(record_dict)
    route_following_stability = max(1 - avg_distance_to_route / 5, 0)
    route_completion = total_route_completion / len(record_dict)
    avg_time_spent = 0 if success_data_cnt == 0 else total_time_spent / success_data_cnt

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

    avg_lane_invasion_freq = num_lane_invasion / len(record_dict)
    avg_acceleration = total_acc / len(record_dict)
    avg_yaw_velocity = total_yaw_velocity / len(record_dict)

    predefined_max_values = {
        # safety level
        'collision_rate': 1,
        'avg_red_light_freq': 1,
        'avg_stop_sign_freq': 1,
        'out_of_road_length': 50,

        # task performance level
        'route_following_stability': 1,
        'route_completion': 1,
        'avg_time_spent': 60,

        # comfort level
        'avg_acceleration': 8,
        'avg_yaw_velocity': 3,
        'avg_lane_invasion_freq': 20,
    }

    scores = {
        # safety level
        'collision_rate': collision_rate,
        'avg_red_light_freq': avg_red_light_freq,
        'avg_stop_sign_freq': avg_stop_sign_freq,
        'out_of_road_length': out_of_road_length,

        # task performance level
        'route_following_stability': route_following_stability,
        'route_completion': route_completion,
        'avg_time_spent': avg_time_spent,

        # comfort level
        'avg_acceleration': avg_acceleration,
        'avg_yaw_velocity': avg_yaw_velocity,
        'avg_lane_invasion_freq': avg_lane_invasion_freq,

        # additional info
        'avg_route_length': avg_route_length,
    }

    # normalized_scores
    ns = {metric: score if metric not in predefined_max_values else score / predefined_max_values[metric] for metric, score in scores.items()}

    final_score = ((1 - ns['collision_rate']) * 5 +
                   (3 - ns['avg_red_light_freq'] - ns['avg_stop_sign_freq'] - ns['out_of_road_length']) * 1 +
                   (ns['route_following_stability'] + ns['route_completion'] + 1 - ns['avg_time_spent']) * 0.5 +
                   (3 - ns['avg_acceleration'] - ns['avg_yaw_velocity'] - ns['avg_lane_invasion_freq']) * 0.2) / 10.1

    ns['safety_os'] = ((1 - ns['collision_rate']) * 5 + (3 - ns['avg_red_light_freq'] - ns['avg_stop_sign_freq'] - ns['out_of_road_length']) * 1) / 8
    ns['task_os'] = (ns['route_following_stability'] + ns['route_completion'] + 1 - ns['avg_time_spent']) * 0.5 / 1.5
    ns['comfort_os'] = (3 - ns['avg_acceleration'] - ns['avg_yaw_velocity'] - ns['avg_lane_invasion_freq']) * 0.2 / 0.6

    return scores, ns, final_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', default='/home/carla/output/testing_records/record.pkl')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    record = joblib.load(args.record_file)
    all_scores, normalized_scores, final_score = get_scores(record)
    print('overall score:', final_score)
