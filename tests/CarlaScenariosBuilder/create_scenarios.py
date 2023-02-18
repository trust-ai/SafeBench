import json

import numpy as np
import os
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

from utilities import select_waypoints, build_scenarios


def draw(ax, center, dist, road_waypoints, selected_waypoints_idx):
    ax.cla()
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y')

    for road_waypoint in road_waypoints:
        length = 4
        x, y, z, pitch, yaw, roll = road_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='y')

    if len(selected_waypoints_idx) > 0:
        waypoints_idx = np.array(selected_waypoints_idx)
        waypoints = np.take(road_waypoints, waypoints_idx, axis=0)

        start_waypoint = waypoints[0]
        for end_waypoint in waypoints[1:]:
            ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
            ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
            ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
        ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
        ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, title=None):
    if title is None:
        title = "Left click:  select or remove waypoints.\nRight click:  save scenarios to file."
    ax.set_title(title, fontsize=25, loc='left')


def rotate_waypoints(origin_waypoints, center, theta):
    # waypoint format: x, y, z, pitch, yaw, roll
    m = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    center = np.array(center).reshape((1, 2))
    rotated_waypoints = origin_waypoints.copy()
    rotated_waypoints[:, :2] = (origin_waypoints[:, :2] - center) @ m.T + center
    # TODO: check if we changed yaw in the correctly
    rotated_waypoints[:, 4] = rotated_waypoints[:, 4] + theta / np.pi * 180
    return rotated_waypoints


def create_scenario_intersection(config, selected_waypoints_idx, road_waypoints):
    # rotate waypoints
    selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
    local_waypoints = []
    if config.multi_rotation:
        for i in range(4):
            theta = i * np.pi / 2
            rotated_waypoints = rotate_waypoints(selected_waypoints, [0, 0], theta)
            local_waypoints.append(rotated_waypoints)
    else:
        local_waypoints.append(selected_waypoints)
    local_waypoints = np.vstack(local_waypoints)

    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, [100, 100], theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_scenario_straight(config, selected_waypoints_idx, road_waypoints):
    # rotate waypoints
    selected_waypoints = np.take(road_waypoints, selected_waypoints_idx, axis=0)
    local_waypoints = []
    if config.multi_rotation:
        for i in range(2):
            theta = i * np.pi
            rotated_waypoints = rotate_waypoints(selected_waypoints, [100, 0], theta)
            local_waypoints.append(rotated_waypoints)
    else:
        local_waypoints.append(selected_waypoints)
    local_waypoints = np.vstack(local_waypoints)

    # rotate waypoints around the map center
    map_waypoints = []
    for i in range(4):
        theta = i * np.pi / 2
        rotated_waypoints = rotate_waypoints(local_waypoints, [100, 100], theta)
        map_waypoints.append(rotated_waypoints)
    map_waypoints = np.vstack(map_waypoints)
    return map_waypoints


def create_scenario(config, selected_waypoints_idx, road_waypoints, waypoints_dense):
    # get point shift in x and y locations
    if config.road == 'intersection':
        all_scenarios_waypoints = create_scenario_intersection(config, selected_waypoints_idx, road_waypoints)
    elif config.road == 'straight':
        all_scenarios_waypoints = create_scenario_straight(config, selected_waypoints_idx, road_waypoints)
    else:
        raise ValueError("--road must be 'intersection' or 'straight'.")

    # save waypoints
    save_dir = os.path.join(config.save_dir, f"scenarios")
    os.makedirs(save_dir, exist_ok=True)

    scenario_num = len(all_scenarios_waypoints) // len(selected_waypoints_idx)
    scenario_length = len(selected_waypoints_idx)

    all_scenarios_configs = []
    for idx in range(scenario_num):
        scenario_waypoints = all_scenarios_waypoints[scenario_length * idx: scenario_length * (idx + 1)]
        real_scenario_waypoints = []

        # find the closest waypoints from the dense waypoints
        for scenario_waypoint in scenario_waypoints:
            waypoints_dist = np.linalg.norm(waypoints_dense[:, :2] - scenario_waypoint[:2].reshape((1, -1)), axis=1)
            idx = waypoints_dist.argmin()
            real_scenario_waypoint = waypoints_dense[idx]
            real_scenario_waypoints.append(real_scenario_waypoint)
            if waypoints_dist.min() > 2:
                print(f"waypoint {scenario_waypoint} can not be found on the map, "
                      f"assigned to the nearist waypoint {real_scenario_waypoint}")

        scenario_config = build_scenarios(real_scenario_waypoints)
        all_scenarios_configs.append(scenario_config)

    # check if we need to create json file
    scenario_id = config.scenario
    save_file = os.path.join(save_dir, f"scenario_{scenario_id:02d}.json")
    if os.path.isfile(save_file):
        with open(save_file, 'r') as f:
            scenario_json = json.load(f)

        if config.map in scenario_json["available_scenarios"][0]:
            scenario_json["available_scenarios"][0][config.map][0][
                "available_event_configurations"] += all_scenarios_configs

    else:
        scenario_json = {
            "available_scenarios": [
                {
                    config.map: [
                        {
                            "available_event_configurations": all_scenarios_configs,
                            "scenario_type": f"Scenario{scenario_id + 2}"
                        }
                    ]
                }
            ]
        }

    with open(save_file, 'w') as f:
        json.dump(scenario_json, f, indent=2)



def main(config):
    # all waypoints on the map
    waypoints_dense = np.load(f"map_waypoints/{config.map}/dense.npy")

    # sparse waypoints used for user to select
    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")

    # waypoints selected by user
    selected_waypoints_idx = []

    # set waypoints that user can work on
    dist = 120
    if config.road == 'intersection':
        center = [0, 0]
    else:
        center = [100, 0]

    def onclick(event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if int(event.button) == 1:
            # left click to add or remove waypoints from the list
            waypoints_dist = np.linalg.norm(road_waypoints[:, :2] - [event.xdata, -event.ydata], axis=1)
            if waypoints_dist.min() < 5:
                idx = waypoints_dist.argmin()
                if idx in selected_waypoints_idx:
                    selected_waypoints_idx.remove(idx)
                else:
                    selected_waypoints_idx.append(idx)

            draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
            set_title(ax)
            plt.draw()

        elif int(event.button) == 3:
            # right click to save waypoints
            if len(selected_waypoints_idx) < 2:
                set_title(ax, "Need at lease 2 waypoints to create a scenario.")
                plt.draw()
            else:
                create_scenario(config, selected_waypoints_idx, road_waypoints, waypoints_dense)
                selected_waypoints_idx.clear()

                draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
                set_title(ax, "Scenario create success! Click to create more scenarios.")
                plt.draw()


    # visualize waypoints
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, center, dist, road_waypoints, selected_waypoints_idx)
    set_title(ax)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=5)
    parser.add_argument('--road', type=str, default='intersection', choices=['intersection', 'straight'],
                        help='Create routes based on a intersection or a straight road.')
    parser.add_argument('--multi_rotation', action='store_true',
                        help='Create multiple symmetrical routes.'
                             'When creating routes that involve an intersection, the code will generate four routes, '
                             'each rotated 90 degrees around the center of the intersection. '
                             'When creating routes alone a straight road, the code will generate two routes, '
                             'each rotated 180 degrees around the center of the road. ')

    args = parser.parse_args()

    main(args)

