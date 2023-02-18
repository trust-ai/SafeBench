import numpy as np
import os
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt


def build_route(waypoints, route_id, town, save_file):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id=f'{route_id}', town=town)

    ET.SubElement(
        route,
        "weather",
        cloudiness="0",
        precipitation="0",
        precipitation_deposits="0",
        wind_intensity="0",
        sun_azimuth_angle="0",
        sun_altitude_angle="70",
        fog_density="0",
        fog_distance="0",
        wetness="0", )

    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        ET.SubElement(
            route,
            "waypoint",
            pitch=f"{pitch:.2f}",
            roll=f"{roll:.2f}",
            x=f"{x:.2f}",
            y=f"{y:.2f}",
            yaw=f"{yaw:.2f}",
            z=f"{z:.2f}"
        )

    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(save_file, encoding='utf-8', xml_declaration=True)


def build_scenarios(waypoints):
    scenario_waypoints = []
    for waypoint in waypoints:
        x, y, z, pitch, yaw, roll = waypoint
        point = {
            "pitch": f'{pitch:.2f}',
            "x": f'{x:.2f}',
            "y": f'{y:.2f}',
            "yaw": f'{yaw:.2f}',
            "z": f'{z:.2f}'
        }
        scenario_waypoints.append(point)

    # The first waypoint is used as the trigger point, rest are used as actor spawn points
    config = {
        "other_actors": {
            "left": scenario_waypoints[1:]
        },
        "transform": scenario_waypoints[0]
    }
    return config


def parse_route(route_file):
    tree = ET.parse(route_file)
    waypoints = []
    for route in tree.iter("route"):
        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            pitch = float(waypoint.attrib['pitch'])
            roll = float(waypoint.attrib['roll'])
            yaw = float(waypoint.attrib['yaw'])
            x = float(waypoint.attrib['x'])
            y = float(waypoint.attrib['y'])
            z = float(waypoint.attrib['z'])
            waypoint_list.append([x, y, z, pitch, yaw, roll])
        waypoints.append(waypoint_list)
    return np.asarray(waypoints)


def parse_scenarios(scenario_config):
    waypoints = []
    trigger_point = scenario_config["transform"]
    x, y, z = float(trigger_point["x"]), float(trigger_point["y"]), float(trigger_point["z"])
    waypoints.append([x, y, z])
    for actor_config in scenario_config["other_actors"]["left"]:
        x, y, z = float(actor_config["x"]), float(actor_config["y"]), float(actor_config["z"])
        waypoints.append([x, y, z])
    return np.asarray(waypoints)


def select_waypoints(waypoints, center, distance):
    # select waypoints around the center within a distance
    mask = np.linalg.norm(waypoints[:, :2] - center, np.inf, axis=1) < distance
    waypoints = waypoints[mask]
    return waypoints



