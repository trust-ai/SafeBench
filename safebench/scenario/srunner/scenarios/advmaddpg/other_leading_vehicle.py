"""
Other Leading Vehicle scenario:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to decelerate.
The ego vehicle has to react accordingly by changing lane to avoid a
collision and follow the leading car in other lane. The scenario ends
either via a timeout, or if the ego vehicle drives some distance.
"""

import carla
import json
import torch
import numpy as np
from .advagent import Agent
from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation
from safebench.scenario.srunner.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.srunner.scenarios.basic_scenario import BasicScenario

from safebench.scenario.srunner.tools.route_manipulation import interpolate_trajectory
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import *

class OtherLeadingVehicle(BasicScenario):

    """
    This class holds everything required for a simple "Other Leading Vehicle"
    scenario involving a user controlled vehicle and two other actors.
    Traffic Scenario 05

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 35
        self._second_vehicle_location = self._first_vehicle_location + 1
        self._ego_vehicle_drive_distance = self._first_vehicle_location * 4
        self._first_vehicle_speed = 12
        self._second_vehicle_speed = 12
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout
        
        self._actor_distance = self._first_vehicle_location
        self.STEERING_MAX=0.3
        self.adv_agent = Agent(chkpt_dir = 'checkpoints/standard_scenario3')
        self.adv_agent.load_models()
        self.adv_agent.eval()
        self.out_lane_thres = 4
        self.max_waypt = 12
        self.routeplanner = None
        self.waypoints = []
        self.vehicle_front = False
        
        self.dece_distance = 5
        self.dece_target_speed = 2  # 3 will be safe

        self.need_decelerate = False

        super(OtherLeadingVehicle, self).__init__("VehicleDeceleratingInMultiLaneSetUp",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

        self.scenario_operation = ScenarioOperation(self.ego_vehicles, self.other_actors)
        self.actor_type_list.append('vehicle.nissan.patrol')
        self.actor_type_list.append('vehicle.audi.tt')
        self.trigger_distance_threshold = 35
        self.other_actor_speed = []
        self.other_actor_speed.append(self._first_vehicle_speed)
        self.other_actor_speed.append(self._second_vehicle_speed)
        self.ego_max_driven_distance = 200

        self.step = 0
        with open(config.parameters, 'r') as f:
            parameters = json.load(f)
        self.control_seq = parameters
        # print(self.control_seq)
        self._other_actor_max_velocity = self.dece_target_speed * 2
        
    def initialize_route_planner(self):
        carla_map = self.world.get_map()
        forward_vector = self.other_actor_transform[0].rotation.get_forward_vector() * self._actor_distance
        self.target_transform = carla.Transform(carla.Location(self.other_actor_transform[0].location + forward_vector),
                                           self.other_actor_transform[0].rotation)
        self.target_waypoint = [self.target_transform.location.x, self.target_transform.location.y, self.target_transform.rotation.yaw]
        print('self.target_waypoint', self.target_waypoint)

        other_locations = [self.other_actor_transform[0].location,
                           carla.Location(self.other_actor_transform[0].location + forward_vector)]
        gps_route, route = interpolate_trajectory(self.world, other_locations)
        init_waypoints = []
        for wp in route:
            init_waypoints.append(carla_map.get_waypoint(wp[0].location))
        
        print('init_waypoints')
        # for i in init_waypoints:
        #     print(i)
        self.routeplanner = RoutePlanner(self.other_actors[0], self.max_waypt, init_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
        
    def initialize_actors(self):
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)

        self.other_actor_transform.append(first_vehicle_transform)
        self.other_actor_transform.append(second_vehicle_transform)
        self.scenario_operation.initialize_vehicle_actors(self.other_actor_transform, self.other_actors, self.actor_type_list)
        # self.reference_actor = self.other_actors[1]
        self.reference_actor = self.other_actors[0]

        self._first_actor_transform = first_vehicle_transform
        # self.second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
        #                                                second_vehicle_waypoint.transform.rotation)
        self.initialize_route_planner()

    def update_behavior(self):
        """
        Just make two vehicles move forward with specific speed
        At specific point, vehicle in front of ego will decelerate
        other_actors[0] is the vehicle before the ego
        """
        # cur_distance = calculate_distance_transforms(CarlaDataProvider.get_transform(self.ego_vehicles[0]),
        #                                              CarlaDataProvider.get_transform(self.other_actors[1]))
        cur_distance = calculate_distance_transforms(self.other_actor_transform[0],
                                                     CarlaDataProvider.get_transform(self.other_actors[0]))

        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
                self.waypoints, _, _, _, _, self.vehicle_front = self.routeplanner.run_step()
                with torch.no_grad():
                    state = self._get_obs()
                    ## eval ##
                    new_state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.adv_agent.actor.device)
                    action = self.adv_agent.actor.forward(new_state).detach().cpu().numpy()[0]
                    ## train ##
        #             action = self.adv_agent.choose_action(state)
                    current_velocity = (action[0]+1)/2 * self._other_actor_max_velocity
                self.scenario_operation.go_straight(current_velocity, 0, steering = action[1] * self.STEERING_MAX)
                self.step += 1
            else:
                self.scenario_operation.go_straight(self.other_actor_speed[i], i)
                                                
    def _get_obs(self):
        self.ego = self.other_actors[0]
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])
        acc = self.ego.get_acceleration()
        acceleration = np.sqrt(acc.x**2 + acc.y**2)
        ### For Prediction, we also need (ego_x, ego_y), ego_yaw, acceleration ###
        state = np.array([
            lateral_dis, -delta_yaw, speed, int(self.vehicle_front),
            (ego_x, ego_y), ego_yaw, acceleration
        ],
                         dtype=object)
        
        ### Relative Postion ###
        ego_location = self.ego_vehicles[0].get_transform().location
        ego_location = np.array([ego_location.x, ego_location.y])

        actor = self.other_actors[0]
        sv_location = actor.get_transform().location
        sv_location = np.array([sv_location.x, sv_location.y])

        rel_ego_location = ego_location - sv_location
        sv_forward_vector = actor.get_transform().rotation.get_forward_vector()
        sv_forward_vector = np.array([sv_forward_vector.x, sv_forward_vector.y])

        projection_x = sum(rel_ego_location * sv_forward_vector)
        projection_y = np.linalg.norm(rel_ego_location - projection_x * sv_forward_vector) * \
                                np.sign(np.cross(sv_forward_vector, rel_ego_location))

        ego_forward_vector = self.ego_vehicles[0].get_transform().rotation.get_forward_vector()
        ego_forward_vector = np.array([ego_forward_vector.x, ego_forward_vector.y])
        rel_ego_yaw = np.arcsin(np.cross(sv_forward_vector, ego_forward_vector))
        
        state = np.concatenate((state[:4], \
                                 np.array([rel_ego_yaw, projection_x, projection_y]))).astype(float)
        return state
                                                
    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.other_actors[0].get_velocity()

        # reward for steering:
        r_steer = -self.other_actors[0].get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.other_actors[0])
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -1

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self._other_actor_max_velocity:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = -abs(self.other_actors[0].get_control().steer) * lspeed_lon**2
        r = 0.1
        r = 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat + 0.1
        
        return r
    
    def _create_behavior(self):
        pass

    def check_stop_condition(self):
        pass






    # def _initialize_actors(self, config):
    #     """
    #     Custom initialization
    #     """
    #     first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
    #     second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
    #     second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
    #
    #     first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
    #                                               first_vehicle_waypoint.transform.rotation)
    #     second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
    #                                                second_vehicle_waypoint.transform.rotation)
    #
    #     first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', first_vehicle_transform)
    #     second_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.tt', second_vehicle_transform)
    #
    #     self.other_actors.append(first_vehicle)
    #     self.other_actors.append(second_vehicle)
    #
    #     self._first_actor_transform = first_vehicle_transform
    #     self._second_actor_transform = second_vehicle_transform
    #
    # def _create_behavior(self):
    #     """
    #     The scenario defined after is a "other leading vehicle" scenario. After
    #     invoking this scenario, the user controlled vehicle has to drive towards the
    #     moving other actors, then make the leading actor to decelerate when user controlled
    #     vehicle is at some close distance. Finally, the user-controlled vehicle has to change
    #     lane to avoid collision and follow other leading actor in other lane to end the scenario.
    #     If this does not happen within 90 seconds, a timeout stops the scenario or the ego vehicle
    #     drives certain distance and stops the scenario.
    #     """
    #     # start condition
    #     driving_in_same_direction = py_trees.composites.Parallel("All actors driving in same direction",
    #                                                              policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #     leading_actor_sequence_behavior = py_trees.composites.Sequence("Decelerating actor sequence behavior")
    #
    #     # both actors moving in same direction
    #     keep_velocity = py_trees.composites.Parallel("Trigger condition for deceleration",
    #                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #     keep_velocity.add_child(WaypointFollower(self.other_actors[0], self._first_vehicle_speed, avoid_collision=True))
    #     keep_velocity.add_child(InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 55))
    #
    #     # Decelerating actor sequence behavior
    #     decelerate = self._first_vehicle_speed / 3.2
    #     leading_actor_sequence_behavior.add_child(keep_velocity)
    #     leading_actor_sequence_behavior.add_child(WaypointFollower(self.other_actors[0], decelerate,
    #                                                                avoid_collision=True))
    #     # end condition
    #     ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_drive_distance)
    #
    #     # Build behavior tree
    #     sequence = py_trees.composites.Sequence("Scenario behavior")
    #     parallel_root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
    #
    #     parallel_root.add_child(ego_drive_distance)
    #     parallel_root.add_child(driving_in_same_direction)
    #     driving_in_same_direction.add_child(leading_actor_sequence_behavior)
    #     driving_in_same_direction.add_child(WaypointFollower(self.other_actors[1], self._second_vehicle_speed,
    #                                                          avoid_collision=True))
    #
    #     sequence.add_child(ActorTransformSetter(self.other_actors[0], self._first_actor_transform))
    #     sequence.add_child(ActorTransformSetter(self.other_actors[1], self._second_actor_transform))
    #     sequence.add_child(parallel_root)
    #     sequence.add_child(ActorDestroy(self.other_actors[0]))
    #     sequence.add_child(ActorDestroy(self.other_actors[1]))
    #
    #     return sequence
    #
    # def _create_test_criteria(self):
    #     """
    #     A list of all test criteria will be created that is later used
    #     in parallel behavior tree.
    #     """
    #     criteria = []
    #
    #     collision_criterion = CollisionTest(self.ego_vehicles[0])
    #     criteria.append(collision_criterion)
    #
    #     return criteria
    #
    # def __del__(self):
    #     self.remove_all_actors()
