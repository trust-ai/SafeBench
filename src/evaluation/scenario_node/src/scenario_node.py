import rospy
import random
import numpy as np
import torch

from carla_msgs.msg import CarlaCollisionEvent
from carla_msgs.msg import CarlaLaneInvasionEvent

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.scenario_parser import ScenarioConfigurationParser

from scenario_runner import ScenarioRunner
from scenario_runner import VERSION

from carla_ros_scenario_runner_types.srv import GetEgoVehicleRoute


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ScenarioArguments:
    def __init__(self,
                 host='127.0.0.1',
                 port=2000,
                 timeout='10.0',
                 trafficManagerPort='8000',
                 trafficManagerSeed='0',
                 sync=False,
                 list=False,
                 scenario=None,
                 openscenario=None,
                 # route_file='',
                 # scenario_file='',
                 agent=None,
                 agentConfig='',
                 output=False,
                 file=False,
                 junit=False,
                 json=False,
                 outputDir='/home/carla/output',
                 configFile='',
                 additionalScenario='',
                 debug=False,
                 reloadWorld=False,
                 record='',
                 randomize=False,
                 repetitions=1,
                 waitForEgo=False,
                 seed=42,

                 data_file='',
                 scenario_id=4,
                 route_id=0,
                 method='',
                 risk_level=0,
                 train_agent=False,
                 train_agent_episodes=100):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.trafficManagerPort = trafficManagerPort
        self.trafficManagerSeed = trafficManagerSeed
        self.sync = sync
        self.list = list
        self.scenario = scenario
        self.openscenario = openscenario
        self.agent = agent
        self.agentConfig = agentConfig
        self.output = output
        self.file = file
        self.junit = junit
        self.json = json
        self.outputDir = outputDir
        self.configFile = configFile
        self.additionalScenario = additionalScenario
        self.debug = debug
        self.reloadWorld = reloadWorld
        self.record = record
        self.randomize = randomize
        self.repetitions = repetitions
        self.waitForEgo = waitForEgo
        self.seed = seed

        self.scenario_id = scenario_id
        self.route_id = route_id
        self.method = method
        self.risk_level = risk_level
        self.train_agent = train_agent
        self.train_agent_episodes = train_agent_episodes

        self.data_file = data_file
        # self.route = []
        # if route_file != '':
        #     self.route.append(route_file)
        #     self.route.append(scenario_file)


def run(arguments):
    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return True

    # if not arguments.scenario and not arguments.openscenario and not arguments.route:
    #     print("Please specify either a scenario or use the route mode\n\n")
    #     return True

    # if arguments.route and (arguments.openscenario or arguments.scenario):
    #     print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
    #     return True

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        return True

    # if arguments.route:
    #     arguments.reloadWorld = False

    if arguments.agent:
        arguments.sync = True

    scenario_runner = None
    result = True
    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()
    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result


if __name__ == "__main__":
    # parameters for ROS
    rospy.init_node("Scenario_Node")
    role_name = rospy.get_param("~role_name", "ego_vehicle")
    data_file = rospy.get_param("~data_file", "")
    # route_file = rospy.get_param("~route_file", "")
    # scenario_file = rospy.get_param("~scenario_file", "")
    train_agent = rospy.get_param("~train_agent", "False")
    train_agent_episodes = rospy.get_param("~train_agent_episodes", "100")
    scenario_id = rospy.get_param("~scenario_id", "04")
    route_id = rospy.get_param("~route_id", "00")
    method = rospy.get_param("~method", "")
    risk_level = rospy.get_param("~risk_level", "0")
    rospy.loginfo("Start generating scenarios for %s using scenario_runner %s" % (role_name, VERSION))

    # for later usage
    rospy.Subscriber('/carla/%s/collision' % role_name, CarlaCollisionEvent, None)
    rospy.Subscriber('carla/%s/lane_invasion' % role_name, CarlaLaneInvasionEvent, None)

    get_ego_vehicle_route_service = rospy.Service('/carla_data_provider/get_ego_vehicle_route',
                                                  GetEgoVehicleRoute, CarlaDataProvider.get_ego_vehicle_route_callback)

    # parameters for carla scenario_runner
    scenario_id = None if scenario_id == -1 else scenario_id
    route_id = None if route_id == -1 else route_id
    method = None if method == '' else method
    risk_level = None if risk_level == -1 else risk_level
    arguments = ScenarioArguments(data_file=data_file,
                                  train_agent=train_agent,
                                  train_agent_episodes=train_agent_episodes,
                                  scenario_id=scenario_id,
                                  route_id=route_id,
                                  method=method,
                                  risk_level=risk_level)

    try:
        set_seed(arguments.seed)
        run(arguments)
    except rospy.exceptions.ROSInterruptException:
        rospy.loginfo("Shutting down scenario node")
