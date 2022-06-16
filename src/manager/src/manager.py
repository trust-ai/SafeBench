""" This is the manager node that process parameters and create agent and scenes.
    This node runs a main loop of the entire platform.
"""
import os
import subprocess
import time
import math
import xml.dom.minidom as xmld
import carla
import carla_common.transforms as trans
from transforms3d.euler import quat2euler

import rospy
import rosnode

from carla_ros_scenario_runner_types.msg import CarlaScenarioRunnerStatus

scenario_status = CarlaScenarioRunnerStatus.RUNNING


four_wheel_vehicle = [
    'vehicle.audi.a2',
    'vehicle.tesla.model3',
    'vehicle.bmw.grandtourer',
    'vehicle.audi.etron',
    'vehicle.seat.leon',
    'vehicle.mustang.mustang',
    'vehicle.tesla.cybertruck',
    'vehicle.lincoln.mkz2017',
    'vehicle.lincoln2020.mkz2020',
    'vehicle.dodge_charger.police',
    'vehicle.audi.tt',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.chevrolet.impala',
    'vehicle.nissan.patrol',
    'vehicle.nissan.micra',
    'vehicle.mercedesccc.mercedesccc',
    'vehicle.mini.cooperst',
    'vehicle.chargercop2020.chargercop2020',
    'vehicle.toyota.prius',
    'vehicle.mercedes-benz.coupe',
    'vehicle.citroen.c3',
    'vehicle.charger2020.charger2020', 
    'vehicle.bmw.isetta'
]


def scenario_runner_status_callback(status):
    global scenario_status
    scenario_status = status.status


class ManagernNode(object):
    def __init__(self, host, port, track, agent_config, data_file, scenario_id, route_id, method, risk_level, train_agent, train_agent_episodes, sample_episode_num, policy, obs_type, load_dir):
        self.host = host 
        self.port = port
        self.track = track 
        self.agent_config = agent_config
        self.data_file = data_file
        self.scenario_id = scenario_id
        self.route_id = route_id
        self.method = method
        self.risk_level = risk_level
        self.train_agent = train_agent
        self.train_agent_episodes = train_agent_episodes
        self.sample_episode_num = sample_episode_num
        self.policy = policy
        self.obs_type = obs_type
        self.load_dir = load_dir

    @staticmethod
    def get_spawn_point(scenario_route_file, route_id=0):
        #spawn_point = [-50.8, -28.06, 2.0, 0.0, 0.0, 78.0]
        spawn_point = None
        dom = xmld.parse(scenario_route_file)
        routes = dom.documentElement.getElementsByTagName('route')
        for r_i in routes:
            if int(r_i.getAttribute('id')) == route_id:
                first_waypoint = r_i.getElementsByTagName('waypoint')[0]
                # print(first_waypoint)
                pitch = float(first_waypoint.getAttribute('pitch'))
                roll = float(first_waypoint.getAttribute('roll'))
                yaw = float(first_waypoint.getAttribute('yaw'))
                x = float(first_waypoint.getAttribute('x'))
                y = float(first_waypoint.getAttribute('y'))
                z = float(first_waypoint.getAttribute('z')) + 2.0  # avoid collision to the ground
                spawn_point = carla.Transform(carla.Location(x, y, z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))
                spawn_point = trans.carla_transform_to_ros_pose(spawn_point)
                roll, pitch, yaw = quat2euler([spawn_point.orientation.w,
                                               spawn_point.orientation.x,
                                               spawn_point.orientation.y,
                                               spawn_point.orientation.z])
                spawn_point = [
                    spawn_point.position.x,
                    spawn_point.position.y,
                    spawn_point.position.z,
                    math.degrees(roll),
                    math.degrees(pitch),
                    math.degrees(yaw)
                ]
                print('ros spawn point', spawn_point)
                # spawn_point = [x, y, z, roll, pitch, yaw]  # "x,y,z,roll,pitch,yaw"
                break

        if spawn_point is None:
            raise ValueError('Invalid route id in route file: {}'.format(scenario_route_file))
        return spawn_point

    def create_agent_and_scene(self):
        v = {}

        # carla ros bridge has a problem spawning vehicle with role_name other than 'ego_vehicle'
        # more information: https://github.com/carla-simulator/ros-bridge/issues/517 
        role_name = 'ego_vehicle'
        v['role_name'] = role_name

        with open(self.agent_config, 'r') as f:
            obj = f.read()
        obj = obj.replace('[[role_name]]', role_name)

        v['track'] = self.track
        vehicle = four_wheel_vehicle[1]  # random.choice(four_wheel_vehicle)
        obj = obj.replace('[[vehicle]]', vehicle)

        json_file = '/tmp/%s.json' % role_name
        v['config_file'] = json_file
        with open(v['config_file'], 'w') as f:
            f.write(obj)

        cmd = 'roslaunch manager create_scenario.launch ' \
              'host:=%s ' \
              'port:=%s ' \
              'role_name:=%s ' \
              'data_file:=%s ' \
              'scenario_id:=%s ' \
              'route_id:=%s ' \
              'method:=%s ' \
              'risk_level:=%s ' \
              'train_agent:=%s ' \
              'train_agent_episodes:=%s &' % \
              (self.host,
              self.port,
              v['role_name'],
              data_file,
              self.scenario_id,
              self.route_id,
              self.method,
              self.risk_level,
              self.train_agent,
              self.train_agent_episodes)
        subprocess.Popen(cmd, preexec_fn=os.setsid, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

        # create agent to control ego vehicles
        #######################################################################################################
        # This part is where you set the agent (algorithm) you use to eval on the benchmark.
        # There are two groups of agents for now:
        # RL agents:
        #       Basic RL (SAC, PPO, DDPG, TD3)
        #       algorithm_name: basic_rl_sac, basic_rl_ppo, basic_rl_ddpg, basic_rl_td3
        # IL agents:
        #       Conditional Imitation Learning (algorithm_name: cilrs)
        #       Learning by Cheating (algorithm_name: lbc)
        #       from Continuous Intention to Continuous Trajectory (algorithm_name: cict)
        #######################################################################################################
        node_name = 'gym_node'
        if node_name == 'gym_node':
            policy = self.policy
            obs_type = self.obs_type
            load_dir = self.load_dir
            if self.train_agent == True:
                mode = 'train'
            else:
                mode = 'eval'
                self.sample_episode_num = 1
            epochs = self.train_agent_episodes / self.sample_episode_num
            sample_episode_num = self.sample_episode_num
        else:
            raise RuntimeError('Wrong agent name.')
        cmd = ('roslaunch '+node_name+' create_agent.launch port:=%s role_name:=%s policy:=%s obs_type:=%s load_dir:=%s mode:=%s epochs:=%s sample_episode_num:=%s &') % tuple([self.port, v['role_name'], policy, obs_type, load_dir, mode, epochs, sample_episode_num])
        subprocess.Popen(cmd, preexec_fn=os.setsid, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    # callback function to rospy.on_shutdown()
    def shut_down(self):
        names = rosnode.get_node_names()
        rosnode.kill_nodes(names)


if __name__ == '__main__':
    rospy.init_node('manager_node')
    host = rospy.get_param('~host', 'localhost')
    port = rospy.get_param('~port', 2000)
    track = rospy.get_param("~track", "Town03")
    agent_config = rospy.get_param("~agent_config", "")
    data_file = rospy.get_param("~data_file", "")
    scenario_id = rospy.get_param("~scenario_id", "04")
    route_id = rospy.get_param("~route_id", "00")
    method = rospy.get_param("~method", "")
    risk_level = rospy.get_param("~risk_level", "0")
    train_agent = rospy.get_param("~train_agent", False)
    train_agent_episodes = rospy.get_param("~train_agent_episodes", "5000")
    sample_episode_num = rospy.get_param("~sample_episode_num", "20")
    policy = rospy.get_param("~policy", "sac")
    obs_type = rospy.get_param("~obs_type", "0")
    load_dir = rospy.get_param("~load_dir", None)

    time.sleep(1)
    try:
        manager_node = ManagernNode(
            host, port, track, agent_config, data_file, scenario_id, route_id, method, risk_level,
            train_agent, train_agent_episodes, sample_episode_num, policy, obs_type, load_dir)
        manager_node.create_agent_and_scene()
        rate = rospy.Rate(20)
        rospy.on_shutdown(manager_node.shut_down)

        status_subscriber = rospy.Subscriber("/scenario_runner/status",
                                             CarlaScenarioRunnerStatus,
                                             scenario_runner_status_callback)
        start_running = False

        # TODO: GUI will be added in this loop
        # TODO: we can make a list to test multiple ego vehicles
        while not rospy.is_shutdown():
            rate.sleep()
            if scenario_status in [CarlaScenarioRunnerStatus.STOPPED, CarlaScenarioRunnerStatus.ERROR, CarlaScenarioRunnerStatus.SHUTTINGDOWN] and start_running:
                for name in rosnode.get_node_names():
                    if name.startswith('gym_node'):
                        continue
                rospy.loginfo("Scenario runner finished, shutting down manager...")
                rospy.signal_shutdown('Scenario runner finished, shutting down manager...')
            elif scenario_status in [CarlaScenarioRunnerStatus.STARTING, CarlaScenarioRunnerStatus.RUNNING] and not start_running:
                start_running = True
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("manager_node shut down exceptionally")
