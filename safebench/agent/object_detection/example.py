from symbol import pass_stmt


import numpy as np


class DummyEgo(object):
    """ This is just an example for testing """
    def __init__(self, config):
        self.action_dim = config['action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'

    def get_action(self, obs):
        # the input should be formed into a batch, the return action should also be a batch
        batch_size = len(obs)
        return np.random.randn(batch_size, self.action_dim)

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode


import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np

DEFAULT_CONFIG = dict(weights=ROOT / 'yolov5n.pt', data=ROOT / 'data/coco128.yaml', \
        imgsz=(1024, 1024), 
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,)

class ObjectDetection(object):
    def __init__(self, config, ) -> None:

        self.action_dim = config['action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.imgsz = DEFAULT_CONFIG['imgsz']
        self.conf_thres = DEFAULT_CONFIG['conf_thres']
        self.iou_thres = DEFAULT_CONFIG['iou_thres']
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.model = DetectMultiBackend(DEFAULT_CONFIG['weights'], device=self.device, dnn=False, data=DEFAULT_CONFIG['data'], fp16=False)
        self.model.warmup(imgsz=(1 if self.model.pt else 1, 3, *DEFAULT_CONFIG['imgsz']))

        stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.annotator = None 
        
        imgsz = check_img_size(DEFAULT_CONFIG['imgsz'], s=stride)  # check image size
    
    def get_action(self, obs):
        print(len(obs), len(obs[0]), type(obs), type(obs[0]))
        image = obs[0][0]['img']
        # print(image.shape)
        image = cv2.resize(image, self.imgsz, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).float().permute(2, 0, 1).to(self.device)
        # print(image.shape)
        image /= 255.
        if len(image.shape) == 3:
            image = image[None]
        pred = self.model(image, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=100)

        # for i, det in enumerate(pred):
        #     if self.annotator is None:
        #         self.annotator = Annotator(image, line_width=3, example=str(self.names))
        
        #     if len(det):
        #         det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image.shape).round()
            
        #     for *xyxy, conf, cls in reversed(det):
        #         c = int(cls)
        #         label = str(self.model.names[c]) + ' {:.2f}'.format(conf)
        #         self.annotator.box_label(xyxy, label, color=colors(c, True)) 
                # print(xyxy, conf)
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        # print(image.shape)
        image = cv2.resize(image, (3000, 3000), interpolation=cv2.INTER_LINEAR)
        image = np.array(255*image, np.uint8)
        # return image
        batch_size = len(obs)
        
        return [{'ego_action': np.array([1.0, 0.0]), 'od_result': pred} for _ in range(batch_size)]

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode
