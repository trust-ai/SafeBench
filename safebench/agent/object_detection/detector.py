'''
Author: Haohong Lin
Email: haohongl@andrew.cmu.edu
Date: 2023-02-04 16:30:08
LastEditTime: 2023-02-12 17:07:50
Description: 
'''

import os
import sys
from pathlib import Path
import yaml
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from safebench.util.od_util import names, CUDA, CPU
from safebench.agent.object_detection.models.common import DetectMultiBackend
from safebench.agent.object_detection.utils.dataloader_label import LoadImagesAndBoxLabels
from safebench.agent.object_detection.utils.loss import ComputeLoss
from safebench.agent.object_detection.utils.plots import Annotator, colors

from safebench.agent.object_detection.utils.general import (
    check_img_size, 
    check_imshow, 
    check_requirements, 
    colorstr, 
    cv2,
    increment_path, 
    non_max_suppression, 
    print_args, 
    scale_coords, 
    strip_optimizer, 
    xyxy2xywh, 
    labels_to_class_weights
)

DEFAULT_CONFIG = dict(weights=ROOT / 'yolov5n.pt', data=ROOT / 'data/coco128.yaml', \
                      imgsz=(1024, 1024),  conf_thres=0.25, iou_thres=0.45,)

# TEMPLATE_DIR = os.path.join()

class ObjectDetection(object):
    def __init__(self, config, logger, train_mode='none') -> None:

        self.ego_action_dim = config['ego_action_dim']
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
        #imgsz = check_img_size(DEFAULT_CONFIG['imgsz'], s=stride)  # check image size
        if train_mode in ['od', 'attack']: 
            self.compute_loss = ComputeLoss(self.model.model)  
        
            # Prepare for training of agents
            self.dataset = LoadImagesAndBoxLabels(path='./online_data/', \
                                            img_size=self.imgsz[0])
            self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True, \
                                        collate_fn=LoadImagesAndBoxLabels.collate_fn)
            
            with open(ROOT / 'data/hyps/hyp.scratch-high.yaml', errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict

            self.model.model.nc = len(names)  # attach number of classes to model
            self.model.model.hyp = hyp  # attach hyperparameters to model
            nl = self.model.model.model[-1].nl
            hyp['box'] *= 3 / nl  # scale to layers
            hyp['cls'] *= 0 # len(names) / 80 * 3 / nl  # scale to classes and layers
            hyp['obj'] *= 0 # (self.imgsz[0] / 640) ** 2 * 3 / nl  # scale to image size and layers
            hyp['label_smoothing'] = 0.0
            self.model.model.class_weights = labels_to_class_weights(self.dataset.labels, len(names)).to('cuda:0') * len(names)  # attach class weights
            self.model.model.names = names

            if train_mode != 'attack': 
                freeze = [0]
                freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
                for k, v in self.model.model.named_parameters():
                    v.requires_grad = True  # train all layers
                    # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
                    if any(x in k for x in freeze):
                        v.requires_grad = False
                
                self.optimizer = self.smart_optimizer(self.model.model, 'Adam', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])    
            
            else:
                self.patch = nn.Parameter(torch.rand(1, 3, 1024, 1024),requires_grad=True)
                self.optimizer = torch.optim.Adam([self.patch], lr=1e-2, betas=(0.9, 0.999))  # adjust beta1 to momentum
   
    
    def get_action(self, obs):
        # print(len(obs), len(obs[0]), type(obs), type(obs[0]))
        image = obs[0]['img']
        # print(image.shape)
        image = cv2.resize(image, self.imgsz, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).float().permute(2, 0, 1).to(self.device)
        # print(image.shape)
        image /= 255.
        if len(image.shape) == 3:
            image = image[None]
        pred = self.model(image, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=100)

        batch_size = len(obs)

        return [{'ego_action': np.array([0.2, 0.0]), 'od_result': pred} for _ in range(batch_size)]

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def train_agent(self, train_mode='od'):
        if train_mode == 'od': 
            self.model.train()
            for e in range(200):
                loss_sum = torch.zeros(3, )
                for img, img0, ret_label in self.train_loader:
                    img = Variable(CUDA(img))
                    ret_label = CUDA(ret_label)
                    pred = self.model.model(img)
                    self.optimizer.zero_grad()
                    loss, loss_items = self.compute_loss(pred, ret_label)
                    loss_items = loss_items.detach()
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += loss_items.cpu()
                loss_sum /= len(self.train_loader)
                print('train_agent: ', loss_sum)
        
        else: # adversarial training
            self.model.train()
            for e in range(200):
                loss_sum = torch.zeros(3, )
                for img, img0, ret_label in self.train_loader:
                    img = CUDA(self.add_patch(CUDA(img), CUDA(self.patch)))
                    ret_label = CUDA(ret_label)
                    pred = self.model.model(img)
                    self.optimizer.zero_grad()
                    loss, loss_items = self.compute_loss(pred, ret_label)
                    loss_items = loss_items.detach()
                    (-loss).backward()
                    self.optimizer.step()
                    loss_sum += loss_items.cpu()
                loss_sum /= len(self.train_loader)
                print('Epoch: ', e, ' | Attack_agents: ', loss_sum)
                if e % 10 == 0: 
                    cv2.imwrite('./online_data/demo_patch.jpg', 255*self.patch.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])
                    cv2.imwrite('./online_data/demo.jpg', 255*img.detach().cpu().numpy().transpose(0, 2, 3, 1)[0])

                    print('saved')

    def smart_optimizer(self, model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            for p_name, p in v.named_parameters(recurse=0):
                if p_name == 'bias':  # bias (no decay)
                    g[2].append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        
        return optimizer

    def annotate(self, pred, img):
        for i, det in enumerate(pred):
            if self.annotator is None:
                self.annotator = Annotator(image, line_width=3, example=str(self.names))
        
            if len(det):
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image.shape).round()
            
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = str(self.model.names[c]) + ' {:.2f}'.format(conf)
                self.annotator.box_label(xyxy, label, color=colors(c, True)) 
        print(xyxy, conf)
        image = image.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        print(image.shape)
        image = cv2.resize(image, self.imgsz, interpolation=cv2.INTER_LINEAR)
        image = np.array(255*image, np.uint8)
        return image
     
    
    def add_patch(self, img, input_patch):
        # img: [1,3,416,416]
        # patch_size = cfg.patch_size
        patch_mask = self.create_patch_mask(img, 512)

        img_mask = self.create_img_mask(img, patch_mask)

        patch_mask = Variable(patch_mask.cuda(), requires_grad=False)
        img_mask = Variable(img_mask.cuda(), requires_grad=False)

        with_patch = img * img_mask + input_patch * patch_mask
        
        return with_patch

    def create_img_mask(self, img, patch_mask):
        mask = torch.ones([3, img.size(2), img.size(3)])
        img_mask = mask - patch_mask

        return img_mask

    def create_patch_mask(self, in_features, patch_size):
        width = in_features.size(2)
        height = in_features.size(3)
        patch_mask = torch.zeros([3, width, height])
        patch_x = 256
        patch_y = 256
        p_w = patch_size + patch_x
        p_h = patch_size + patch_y
        patch_mask[:, int(patch_x):int(p_w), int(patch_y):int(p_h)]= 1

        return patch_mask
    
if __name__ == '__main__':
    agent = ObjectDetection({'ego_action_dim': 2, 'model_path': None}, None, train_mode='attack')
    agent.train_agent('attack')