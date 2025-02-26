# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmengine.config import Config


#frj
from core.models.frelayer import LFE 
import torch.nn.functional as F
from core.models.fdlnet_deeplab import SFF



@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        #frj
        # self.lfe0 = LFE(channel=384, dct_h=8, dct_w=8, frenum=8)  # 假设 in_channels=384
        # self.lfe1 = LFE(channel=768, dct_h=8, dct_w=8, frenum=8)  # 根据实际特征层通道数调整
        # self.lfe2 = LFE(channel=1536, dct_h=8, dct_w=8, frenum=8)
        # self.sff0 = SFF(in_channels=384)
        # self.sff1 = SFF(in_channels=768)
        # self.sff2 = SFF(in_channels=1536)
        #frj
        

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # results = self.bbox_head.forward(img_feats, txt_feats)
        # return results
        
        return img_feats
    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        print("batch_inputsshape:{}".format(batch_inputs.shape))
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        # print("img_backbone_out_shape:{}".format(img_feats[0].shape))
        # print("text_backbone_out_shape:{}".format(txt_feats.shape))
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        #print("img_neck_out_shape:{}".format(img_feats[0].shape))
        # print("text_neck_out_shape:{}".format(txt_feats.shape))
        #img_feats=self.process_features(img_feats)
        #print("img_ftt_out_shape:{}".format(img_feats[0].shape))
        return img_feats, txt_feats
    
    # def process_features(self,spatial_feat, in_channels=384, frenum=8):
    #     """
    #     使用频域特征增强模块处理输入的空间特征。

    #     参数:
    #     - spatial_feat: 空间特征张量，形状为 (batch_size, in_channels, height, width)
    #     - in_channels: 输入特征的通道数，默认为512
    #     - frenum: 频域分量数目，默认为8

    #     返回:
    #     - fused_feat: 融合后的特征，形状与输入空间特征相同
    #     """
    #     #1. 频域特征提取模块
    #     print("fusion_in_channel:{}".format(in_channels))
        
    #     # Step1: 提取频域特征
    #     freq_feat0 = self.lfe0(spatial_feat[0])
    #     freq_feat1 = self.lfe1(spatial_feat[1])
    #     freq_feat2 = self.lfe2(spatial_feat[2])
    #     print("fft succeed")
    #     print("freq_feat0:{}".format(freq_feat0.shape))
    #     print("freq_feat1:{}".format(freq_feat1.shape))
    #     print("freq_feat2:{}".format(freq_feat2.shape))
    #     # Step2: 扩展频域特征的形状以匹配空间特征
    #     fused_feat0 = self.sff0(spatial_feat[0], freq_feat0.expand_as(spatial_feat[0]))
    #     fused_feat1 = self.sff1(spatial_feat[1], freq_feat1.expand_as(spatial_feat[1]))
    #     fused_feat2 = self.sff2(spatial_feat[2], freq_feat2.expand_as(spatial_feat[2]))
    #     return (fused_feat0,fused_feat1,fused_feat2)


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats

#mp,frj,新代码:
def build_backbone(
    image_model_type: str = 'YOLOv8CSPDarknet',
    image_model_config: dict = None,
    text_model_name: str = '/home/ma-user/work/ymxwork/NIPS/YOLO-World/third_party/clip-vit-base-patch32',
    frozen_stages: int = 0,
    text_frozen_modules: List[str] = ['all']
) -> nn.Module:
    """构建多模态Backbone
    
    Args:
        image_model_type (str): 图像Backbone类型，默认YOLOv8CSPDarknet
        image_model_config (dict): 图像Backbone配置参数
        text_model_name (str): 文本模型名称，默认CLIP-ViT-Base
        frozen_stages (int): 冻结的图像Backbone阶段数
        text_frozen_modules (list): 冻结的文本模块列表
        
    Returns:
        MultiModalYOLOBackbone实例
    """
    # 图像Backbone默认配置
    default_image_cfg = dict(
        type=image_model_type,
        deepen_factor=2.5,
        widen_factor=2.5,
        last_stage_out_channels=1024,
        # norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        # act_cfg=dict(type='SiLU', inplace=True)
    )
    if image_model_config:
        default_image_cfg.update(image_model_config)
    
    # 文本Backbone配置
    text_cfg = dict(
        type='mmyolo.HuggingCLIPLanguageBackbone',
        model_name=text_model_name,
        frozen_modules=text_frozen_modules
    )
    
    # 组合多模态Backbone
    backbone_cfg = dict(
        type='mmyolo.MultiModalYOLOBackbone',
        image_model=default_image_cfg,
        text_model=text_cfg,
        frozen_stages=frozen_stages
    )
    
    # return MODELS.build(Config(backbone_cfg))
    return backbone_cfg

def build_neck(
    in_channels: List[int]=[256, 512, 1024],
    guide_channels: int = 512,
    embed_channels: Union[List[int], int] = [128,256,512],
    num_heads: Union[List[int], int] = [16, 32, 64],
    block_type: str = 'mmyolo.MaxSigmoidCSPLayerWithTwoConv',
    freeze_all: bool = True
) -> nn.Module:
    """构建YOLO-World专用Neck
    
    Args:
        in_channels (list): 输入通道数，如[256, 512, 512]
        guide_channels (int): 文本引导通道数
        embed_channels (int/list): 特征嵌入通道数
        num_heads (int/list): 注意力头数
        block_type (str): 构建块类型，默认MaxSigmoidCSP
        freeze_all (bool): 是否冻结所有Neck参数
        示例：
        in_channels: List[int]=[256, 512, 1024],
        guide_channels: int = 512,
        embed_channels: Union[List[int], int] = [128, 256, 512],
        num_heads: Union[List[int], int] = [8, 16, 32]

    Returns:
        YOLOWorldDualPAFPN实例
    """
    # 统一输入格式
    if isinstance(embed_channels, int):
        embed_channels = [embed_channels]*3
    if isinstance(num_heads, int):
        num_heads = [num_heads]*3
    # 构建配置字典
    # neck_cfg = dict(
    #     type='mmyolo.YOLOWorldDualPAFPN',
    #     in_channels=in_channels,
    #     guide_channels=guide_channels,
    #     embed_channels=embed_channels,
    #     num_heads=num_heads,
    #     freeze_all=freeze_all,
    #     block_cfg=dict(type=block_type),
    #     text_enhancer=dict(
    #         type='mmyolo.ImagePoolingAttentionModule',
    #         embed_channels=256,
    #         num_heads=8
    #     )
    # )
    neck_cfg=dict(
        type='mmyolo.YOLOWorldDualPAFPN',
        deepen_factor=2.5,
        widen_factor=2.5,
        in_channels=in_channels,
        guide_channels=guide_channels,
        embed_channels=embed_channels,
        num_heads=num_heads,
        freeze_all=freeze_all,
        block_cfg=dict(type=block_type),
        out_channels=[256, 512, 1024])
    neck = MODELS.build(Config(neck_cfg))
    if freeze_all:
        for param in neck.parameters():
            param.requires_grad = False
    #return neck
    print(neck_cfg)
    return neck_cfg

def build_bboxhead():
    bbox_head=dict(type='mmyolo.YOLOWorldSegHead',
                   head_module=dict(type='mmyolo.YOLOWorldSegHeadModule',
                                    widen_factor=2.0,
                                    in_channels=[256, 256, 512],
                                    embed_dims=512,
                                    num_classes=19,
                                    mask_channels=19,
                                    proto_channels=19,
                                    freeze_bbox=True),
                   mask_overlap=1,
                   loss_mask=dict(type='mmdet.CrossEntropyLoss',
                                  use_sigmoid=True,
                                  reduction='none'),
                   loss_mask_weight=1.0)
    return bbox_head

def build_yoloworld():
    backbone = build_backbone()
    neck = build_neck()
    bbohead=build_bboxhead()
    model = YOLOWorldDetector(
        backbone = backbone,
        neck = neck, 
        bbox_head=bbohead,
        mm_neck=True,
        num_train_classes=19,
        num_test_classes=19
    )
    return model

#mp,frj:

# _base_ = (
#     '/home/ma-user/work/ymxwork/NIPS/YOLO-World/third_party/mmyolo/configs/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'
# )

# custom_imports = dict(imports=['/home/ma-user/work/ymxwork/NIPS/YOLO-World/yolo_world'],
#                       allow_failed_imports=False)

# #frj
# deepen_factor = 1.00
# widen_factor = 1.00
# last_stage_out_channels = 512

# mixup_prob = 0.15
# copypaste_prob = 0.3

# # =======================Unmodified in most cases==================
# # img_scale = _base_.img_scale
# # pre_transform = _base_.pre_transform
# # last_transform = _base_.last_transform
# # affine_scale = _base_.affine_scale

# models = dict(
#     backbone=dict(
#         type='mmyolo.YOLOv8CSPDarknet',
#         last_stage_out_channels=last_stage_out_channels,
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor),
#     neck=dict(
#         type='mmyolo.YOLOv8PAFPN',
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, last_stage_out_channels],
#         out_channels=[256, 512, last_stage_out_channels]),
#     bbox_head=dict(
#         type='mmyolo.YOLOv8Head',
#         head_module=dict(
#             widen_factor=widen_factor,
#             in_channels=[256, 512, last_stage_out_channels])))
# #frj
# def build_yoloworld(args):
#     # 创建YOLOWorld模型
#     #MODELS.register_module(module=YOLOWDetDataPreprocessor)
#     #print("已注册的模型类：", list(MODELS.module_dict.keys()))

#     model_config = dict(
#         type='mmyolo.YOLOWorldDetector',
#         mm_neck=True,
#         num_train_classes=args.num_training_classes,
#         num_test_classes=args.num_test_classes,
#         data_preprocessor=dict(type='mmyolo.YOLOWDetDataPreprocessor'),
#         #data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
#         backbone=dict(
#             #_delete_=True,
#             type='mmyolo.MultiModalYOLOBackbone',
#             #frj
#             #image_model={{_base_.model.backbone}},
            
#             image_model=models["backbone"],
#             frozen_stages=4,  # 冻结图像部分的backbone
#             text_model=dict(
#                 type='mmyolo.HuggingCLIPLanguageBackbone',
#                 model_name='/home/ma-user/work/ymxwork/NIPS/YOLO-World/third_party/clip-vit-base-patch32',
#                 frozen_modules=['all']  # 冻结text model的所有层
#             )
#         ),
#         neck=dict(
#             type='mmyolo.YOLOWorldDualPAFPN',
#             freeze_all=True,
#             guide_channels=args.text_channels,
#             embed_channels=args.neck_embed_channels,
#             num_heads=args.neck_num_heads,
#             block_cfg=dict(type='mmyolo.MaxSigmoidCSPLayerWithTwoConv'),
#             text_enhancer=dict(
#                 type='mmyolo.ImagePoolingAttentionModule',
#                 embed_channels=256,
#                 num_heads=8
#             )
#         ),
#         bbox_head=dict(
#             type='mmyolo.YOLOWorldSegHead',
#             head_module=dict(
#                 type='mmyolo.YOLOWorldSegHeadModule',
#                 embed_dims=args.text_channels,
#                 num_classes=args.num_training_classes,
#                 mask_channels=32,
#                 proto_channels=256,
#                 freeze_bbox=True
#             ),
#             mask_overlap=args.mask_overlap,
#             loss_mask=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=True,
#                 reduction='none'
#             ),
#             loss_mask_weight=1.0
#         ),
#         train_cfg=dict(assigner=dict(num_classes=args.num_training_classes)),
#         test_cfg=dict(mask_thr_binary=0.5, fast_test=True)
#     )

#     print("====================================================")
#     print(model_config)
#     print("==================== ================================")
#     model = MODELS.build(model_config)
#     print("====================================================")
#     print(model)
#     print("====================================================")
#     return model

