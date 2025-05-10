"""Model store which handles pretrained models """

import torch
from collections import OrderedDict
from .fdlnet_deeplab import *
from core.models.yolo_world.detectors.yolo_world import build_yoloworld
from core.models.Segmentatioon import YOLOWDETRsegm

__all__ = ['get_segmentation_model']

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  
        new_state_dict[k] = v
    return new_state_dict

def get_yoloworld(models, model):
    model = models[model]()

    model = YOLOWDETRsegm(model)

    return model

def get_segmentation_model(model, **kwargs):
    models = {
        'fdlnet': get_fdlnet,
        'yoloworld': build_yoloworld,
    }
    if model == 'yoloworld':
        return get_yoloworld(models, model)
    else:
        return models[model](**kwargs)
