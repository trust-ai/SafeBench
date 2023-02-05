from __future__ import print_function
import carla
from safebench.scenario.srunner.tools.scenario_operation import ScenarioOperation
from safebench.scenario.srunner.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.srunner.scenario_dynamic.basic_scenario_dynamic import BasicScenarioDynamic
from safebench.scenario.srunner.scenario_dynamic.route_scenario_dynamic import RouteScenarioDynamic
from safebench.scenario.srunner.scenario_dynamic.route_scenario_dynamic import *

import numpy as np
import os
import cv2


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


class ObjectDetectionDynamic(RouteScenarioDynamic):
    """
    This class creates scenario where ego vehicle 
    is required to conduct pass-by testing.
    """

    def __init__(self, world, config, ego_id, ROOT_DIR, criteria_enable=True):
        '''
        self.world = world
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
        '''

        TEMPLATE_DIR = os.path.join(ROOT_DIR, 'safebench/scenario/scenario_data/template_od')
        self.object_dict = dict(stopsign=['BP_Stop_2'], car=['SM_JeepWranglerRubicon_6', 'SM_Tesla_11'])
        self.image_path_list = [os.path.join(TEMPLATE_DIR, k)+'.jpg' for k in self.object_dict.keys()]

        self.image_list = [cv2.imread(image_file) for image_file in self.image_path_list]
        self.image_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.image_list]
        
        self.bbox_ground_truth = {}
        self.ground_truth_bbox = {}

        super(ObjectDetectionDynamic, self).__init__(world, config, ego_id, criteria_enable)

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
    
    '''
    def initialize_actors(self):
        def find_weather_presets():
            rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
            name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
            presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
            return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
        
        _weather_presets = find_weather_presets()
        idx = 78 # 17
        bp = self.world.get_blueprint_library().filter('tesla')[0]
        bp.set_attribute('role_name', 'hero')
        self.world.spawn_actor()

        self.ego_vehicles = self.world.spawn_actor(bp, self.world.get_map().get_spawn_points()[idx])
        self.ego_vehicles.set_autopilot(True)

        # turn on the light
        light_state = carla.VehicleLightState(carla.VehicleLightState.All)
        for actor in self.world.get_actors():
            if actor.type_id.startswith("vehicle"):
                actor.set_light_state(light_state)
    '''
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

        self.K = build_projection_matrix(image_w, image_h, fov)
        self.bbox_ground_truth['stopsign'] = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        self.bbox_ground_truth['car'] = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        self.bbox_ground_truth['person'] = self.world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        label_verts = [2, 3, 6, 7]
        bbox_label = []

        for key, bounding_box_set in self.bbox_ground_truth.items():
            for bbox in bounding_box_set:
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
    
    def __del__(self):
        return super().__del__()
