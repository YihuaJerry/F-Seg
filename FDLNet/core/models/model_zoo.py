"""Model store which handles pretrained models """

#frj
import torch
from collections import OrderedDict
from .fdlnet_deeplab import *
from core.utils.slconfig import SLConfig
#frj

# from .fdlnet_psp import * 

__all__ = ['get_segmentation_model']

# def get_segmentation_model(model, **kwargs):
#     models = {
#         'fdlnet': get_fdlnet
#     }
#     return models[model](**kwargs)

#mp:
from FDLNet.core.models.yolo_world.detectors.yolo_world import build_yoloworld
from FDLNet.core.models.DarkDino_Segmentatioon import YOLOWDETRsegm

#frj
def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
#frj


def get_yoloworld(models, model):
    # args = SLConfig.fromfile("/home/ma-user/work/ymxwork/NIPS/YOLO-World/configs/5_final_config.py")
    # args.device = "cuda"
    device = torch.device(0)
    model = models[model]()
    
    #checkpoint = torch.load('/home/ma-user/work/ymxwork/NIPS/YOLO-World/pretrained_weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth', map_location=device)
    #load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    #model.load_state_dict(checkpoint['state_dict'])
    model = YOLOWDETRsegm(model)
    
    #加载预训练权重
    
    # checkpoint=torch.load('/home/ma-user/work/ymxwork/NIPS/YOLO-World/FDLNet/runs/ckpt/last_yoloworld_resnet50_night_epoch_260_mean_iu_0.44527.pth',map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])

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
