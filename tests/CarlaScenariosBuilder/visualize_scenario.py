import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utilities import parse_scenarios, select_waypoints


def draw(ax, scenario_config, centers, waypoints_sparse, zoom):
    if zoom:
        draw_zoom(ax, scenario_config, centers, waypoints_sparse)
    else:
        draw_global(ax, scenario_config, waypoints_sparse)


def draw_global(ax, scenario_config, waypoints_sparse):
    ax.cla()
    ax.plot(waypoints_sparse[:, 0], -waypoints_sparse[:, 1], 'o', color='y')

    waypoints = parse_scenarios(scenario_config)
    print(waypoints)

    start_waypoint = waypoints[0]
    for end_waypoint in waypoints[1:]:
        print(end_waypoint)
        ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
        ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
        ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
    ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
    ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    center = [100, 100]
    dist = 220
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def draw_zoom(ax, scenario_config, centers, waypoints_sparse):

    waypoints = parse_scenarios(scenario_config)

    geo_center = waypoints[:, :2].mean(0)
    center_dists = np.linalg.norm(np.array(centers) - geo_center, axis=1)
    center = centers[center_dists.argmin()]
    dist = 120

    ax.cla()
    road_waypoints = select_waypoints(waypoints_sparse, center, dist)
    ax.plot(road_waypoints[:, 0], -road_waypoints[:, 1], 'o', color='y')
    for road_waypoint in road_waypoints:
        length = 4
        x, y, z, pitch, yaw, roll = road_waypoint
        yaw = yaw / 180 * np.pi
        dx, dy = length * np.cos(yaw), length * np.sin(yaw)
        plt.arrow(x, -y, dx, -dy, color='y')

    start_waypoint = waypoints[0]
    for end_waypoint in waypoints[1:]:
        ax.plot([start_waypoint[0], end_waypoint[0]], [-start_waypoint[1], -end_waypoint[1]], '--', color='b')
        ax.plot(end_waypoint[0], -end_waypoint[1], 'o', color='g')
        ax.text(end_waypoint[0] + 8, -end_waypoint[1] + 8, "Actor", bbox=dict(facecolor='green', alpha=0.7))
    ax.plot(start_waypoint[0], -start_waypoint[1], 'o', color='r')
    ax.text(start_waypoint[0] + 12, -start_waypoint[1] + 8, "Trigger", bbox=dict(facecolor='red', alpha=0.7))

    dist = 120
    x_min, x_max = center[0] - dist, center[0] + dist
    y_min, y_max = -center[1] - dist, -center[1] + dist
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def set_title(ax, route_file, title=None):
    if title is None:
        title = "Left and right click to change route.\nMiddle click to zoom."
    title = f"Visualizing route: {route_file}.\n" + title
    ax.set_title(title, fontsize=12, loc='left')


def main(config):
    # sparse waypoints used for visualization
    waypoints_sparse = np.load(f"map_waypoints/{config.map}/sparse.npy")

    # initialize potential center position
    centers = [
        [100, 0],
        [200, 100],
        [100, 200],
        [0, 100]
    ]
    for i in range(2):
        for j in range(2):
            centers.append([200 * i, 200 * j])

    # get scenarios
    scenario_id = config.scenario
    scenario_dir = os.path.join(config.save_dir, f"scenarios/scenario_{scenario_id:02d}.json")
    with open(scenario_dir, 'r') as f:
        scenario_json = json.load(f)

    scenario_configs = scenario_json["available_scenarios"][0][config.map][0]["available_event_configurations"]
    scenario_num = len(scenario_configs)

    scenario_idx = 0
    zoom = False

    assert scenario_num > 0, "No scenario to visualize."

    def onclick(event):
        nonlocal scenario_idx, zoom

        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))

        new_scenario_id = scenario_idx
        if int(event.button) == 1:
            # left click to visualize next scenario
            new_scenario_id = (scenario_idx + 1) % scenario_num
        elif int(event.button) == 3:
            # right click to visualize previous scenario
            new_scenario_id = (scenario_idx - 1) % scenario_num
        elif int(event.button) == 2:
            # middle click to zoom
            zoom = not zoom

        if new_scenario_id != scenario_idx:
            scenario_idx = new_scenario_id

        if 1 <= int(event.button) <= 3:
            draw(ax, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom)
            set_title(ax, scenario_idx)
            plt.draw()

    # visualize waypoints
    fig, ax = plt.subplots(figsize=(12, 12))
    draw(ax, scenario_configs[scenario_idx], centers, waypoints_sparse, zoom)
    set_title(ax, scenario_idx)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Town_Safebench')
    parser.add_argument('--save_dir', type=str, default="scenario_data/route_new_map")
    parser.add_argument('--scenario', type=int, default=5)

    args = parser.parse_args()

    main(args)
