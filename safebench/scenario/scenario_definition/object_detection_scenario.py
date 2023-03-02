'''
Author:
Email: 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-02 16:37:19
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os

import carla
import numpy as np
import cv2

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_definition.route_scenario import *
from safebench.util.od_util import *


class ObjectDetectionScenario(BasicScenario):
    """
    This class creates scenario where ego vehicle 
    is required to conduct pass-by testing.
    """

    def __init__(self, world, config, ego_id, ROOT_DIR, logger, criteria_enable=True, first_env=False):
        self.world = world
        self.logger = logger
        self.config = config
        self.route = None
        self.ego_id = ego_id
        self.sampled_scenarios_definitions = None
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self._update_route(world, config)
        ego_vehicle = self._update_ego_vehicle()
        self.list_scenarios = self._build_scenario_instances(
            world,
            ego_vehicle,
            self.sampled_scenarios_definitions,
            scenarios_per_tick=5,
            timeout=self.timeout,
            weather=config.weather
        )
        self.n_step = 0

        TEMPLATE_DIR = os.path.join(ROOT_DIR, 'safebench/scenario/scenario_data/template_od')
        self.object_dict = dict(
            stopsign=list(filter(lambda k: 'BP_Stop' in k, world.get_names_of_all_objects())),
            car=list(filter(lambda k: 'SM_Tesla' in k or 'SM_Jeep' in k, world.get_names_of_all_objects())),
            ad=list(filter(lambda k: 'AD' in k, world.get_names_of_all_objects()))
        )
        self.image_path_list = [os.path.join(TEMPLATE_DIR, k)+'.jpg' for k in self.object_dict.keys()]

        self.image_list = [cv2.imread(image_file) for image_file in self.image_path_list]
        self.image_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.image_list]
        
        self.bbox_ground_truth = {}
        self.ground_truth_bbox = {}
        

        super(ObjectDetectionScenario, self).__init__(
            name=config.name,
            ego_vehicles=[ego_vehicle],
            config=config,
            world=world,
            debug_mode=False,
            terminate_on_failure=False,
            criteria_enable=criteria_enable, 
            first_env=first_env
        )
        self.criteria = self._create_criteria()
        # TODO: make save dir, add flag
        os.makedirs('online_data/', exist_ok=True)
        os.makedirs('online_data/images', exist_ok=True)
        os.makedirs('online_data/labels', exist_ok=True)
        self.video_writer = xverse_video_writer('online_data/images/debug.mp4', 1024, 1024)

    def _initialize_environment(self, world): # TODO: image from dict or parameter?
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        for obj_key, image in zip(self.object_dict.keys(), self.image_list):
            resized = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_AREA)
            resized = np.rot90(resized,k=1)
            resized = cv2.flip(resized,1)
            height = 1024
            texture = carla.TextureColor(height,height)
            for x in range(1024):
                for y in range(1024):
                    r = int(resized[x,y,0])
                    g = int(resized[x,y,1])
                    b = int(resized[x,y,2])
                    a = int(255)
                    # texture.set(x,height -0-y - 1, carla.Color(r,g,b,a))
                    texture.set(height-x-1, height-y-1, carla.Color(r,g,b,a))
                    # texture.set(x, y, carla.Color(r,g,b,a))
            for o_name in self.object_dict[obj_key]:
                world.apply_color_texture_to_object(o_name, carla.MaterialParameter.Diffuse, texture)
    
    def initialize_actors(self):
        return

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicles[0]),
            'ego_acceleration_x': self.ego_vehicles[0].get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicles[0].get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicles[0].get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicles[0]).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicles[0]).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Stop due to collision', color='yellow')
        if self.route_length > 1:  # only check when evaluating
            #print(running_status['route_complete'])
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Stop due to route complete', color='yellow')
            if running_status['speed_above_threshold'] == Status.FAILURE:
                if running_status['route_complete'] == 0:
                    raise RuntimeError("Agent not moving")
                else:
                    stop = True
                    self.logger.log('>> Stop due to low speed', color='yellow')
        else:
            if len(running_record) >= 250:  # stop at max step when training
                stop = True
                self.logger.log('>> Stop due to max step', color='yellow')

        for scenario in self.list_scenarios:
            if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                stop = True
                self.logger.log('>> Stop due to max driven distance', color='yellow')
                break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Stop due to timeout', color='yellow')
                break
        
        return running_status, stop

    def _update_route(self, world, config, timeout=None):
        """
            Update the input route, i.e. refine waypoint list, and extract possible scenario locations

            Parameters:
            - world: CARLA world
            - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        if config.scenario_file is not None:
            world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        else:
            world_annotations = config.scenario_config

        # prepare route's trajectory (interpolate and add the GPS route)
        len_trajectory = len(config.trajectory)
        # print(f"length of trajectory {len_trajectory}")
        if len_trajectory == 0:
            len_spawn_points = len(self.vehicle_spawn_points)
            idx = random.choice(list(range(len_spawn_points)))
            random_transform = self.vehicle_spawn_points[idx]
            route = interpolate_trajectory(world, [random_transform])
        else:
            route = interpolate_trajectory(world, config.trajectory)

        potential_scenarios_definitions, _, t, mt = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)

        self.route = route
        self.route_length = len(route)
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        CarlaDataProvider.set_scenario_config(config)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout() if timeout is None else timeout

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rng = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters
        min_length = 100.0

        if len(self.route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]

        # gradually increase the height of ego vehicle
        success = False
        while not success:
            try:
                role_name = 'ego_vehicle'+str(self.ego_id)
                ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', elevate_transform, rolename=role_name)
                if ego_vehicle is not None:
                    success = True
                else:
                    elevate_transform.location.z += 0.1
            except RuntimeError:
                elevate_transform.location.z += 0.1

        return ego_vehicle

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions, scenarios_per_tick=5, timeout=300, weather=None):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []
        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            scenario_class = NUMBER_CLASS_TRANSLATION[self.config.scenario_generation_method][definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.parameters = self.config.parameters

            if weather is not None:
                scenario_configuration.weather = weather

            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.tesla.model3', ego_vehicle.get_transform(), 'ego_vehicle')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name

            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration, criteria_enable=False, timeout=timeout)
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                scenario_number += 1
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        We need this method for background
        We keep pytrees just for background
        """
        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SuccessOnOne)
        subbehavior = py_trees.composites.Parallel(name="Behavior", policy=py_trees.common.ParallelPolicy.SuccessOnAll)

        # subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)
        return behavior

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)
        
        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicles[0], distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicles[0], avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicles[0], optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicles[0], optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicles[0], terminate_on_failure=True)
        # criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicles[0])
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicles[0])
        if self.route_length > 1:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicles[0], route=route, offroad_max=30)
            criteria['speed_above_threshold'] = ActorSpeedAboveThresholdTest(
                actor=self.ego_vehicles[0],
                speed_threshold=0.1,
                below_threshold_max_time=10,
                terminate_on_failure=True
            )
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicles[0], route=route)
        return criteria

    def get_bbox(self, world_2_camera, image_w, image_h, fov): 
        def get_image_point(loc, K, w2c):
            # Calculate 2D projection of 3D coordinate

            # Format the input coordinate (loc is a carla.Position object)
            point = np.array([loc.x, loc.y, loc.z, 1])
            # transform to camera coordinates
            point_camera = np.dot(w2c, point)

            # New we must change from UE4's coordinate system to an "standard"
            # (x, y ,z) -> (y, -z, x)
            # and we remove the fourth componebonent also
            point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

            # now project 3D->2D using the camera matrix
            point_img = np.dot(K, point_camera)
            # normalize
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

            return point_img[0:2]

        self.ground_truth_bbox = {}
        self.K = build_projection_matrix(image_w, image_h, fov)
        self.bbox_ground_truth['stopsign'] = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        self.bbox_ground_truth['car'] = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        self.bbox_ground_truth['person'] = self.world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        label_verts = [2, 3, 6, 7]

        for key, bounding_box_set in self.bbox_ground_truth.items():
            for bbox in bounding_box_set:
                bbox_label = []
                if bbox.location.distance(self.ego_vehicles[0].get_transform().location) < 50:
                    forward_vec = self.ego_vehicles[0].get_transform().get_forward_vector()
                    ray = bbox.location - self.ego_vehicles[0].get_transform().location
                    if forward_vec.dot(ray) > 5.0:
                        verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                        
                        for v in label_verts:
                            p = get_image_point(verts[v], self.K, world_2_camera)
                            bbox_label.append(p)
                        
                        self.ground_truth_bbox.setdefault(key, [])
                        self.ground_truth_bbox[key].append(np.array(bbox_label))
    
    def eval(self, bbox_pred, bbox_gt):
        # print(bbox_pred)
        # print(bbox_gt)
        types = bbox_pred[0][:, -1].detach().cpu().numpy()
        types = np.array(types, dtype=np.int32)
        pred = bbox_pred[0][:, :-2]
        # pred = xywh2xyxy(pred)
        # print(pred.shape)
        # print(bbox_gt.keys())
        for obj_idx in range(len(types)):
            if names[types[obj_idx]] in bbox_gt.keys():
                box_true = bbox_gt[names[types[obj_idx]]]

                if len(box_true) > 0:
                    box_pred = pred[obj_idx]
                    for b_true in box_true:
                        b_true = get_xyxy(b_true)
                        ret = box_iou(box_pred[None, :].detach().cpu(), b_true[None, :])[0][0]
                        if ret > 0:
                            print('detected: ', names[types[obj_idx]], '|  IoU: ', ret.item())
                else:
                    continue
    
    def save_img_label(self, obs, label):
        self.video_writer.write_frame(obs)

        saved_list = []
        for k in label.keys():
            cls = names.index(k)
            bbox_true = label[k]
            for box_true in bbox_true:
                box_save = xyxy2xywhn(get_xyxy(box_true)[None, :])[0]
                saved_list.append(np.concatenate([np.array([cls]), box_save.numpy()], axis=0))
        # print(self.n_step, obs.shape)
        # save_image('online_data/images/'+str(self.n_step)+'.jpg', obs)
        print('saved: ', self.n_step)
        np.savetxt('online_data/labels/'+str(self.n_step)+'.txt', np.array(saved_list), delimiter=' ')
        self.n_step += 1
        pass

    def __del__(self):
        self.video_writer.release()
        for scenario in self.list_scenarios:
            scenario.remove_all_actors()
        self.remove_all_actors()
        