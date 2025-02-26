# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmengine.config import Config

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
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
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
            print("0------------------------------------------------------")
            print('batch_input.shape:{}'.format(batch_inputs.shape))
            print("0------------------------------------------------------")
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
            print("1------------------------------------------------------")
            print('backbone.shape0:{}'.format(img_feats[0].shape))
            print('backbone.shape1:{}'.format(img_feats[1].shape))
            print('backbone.shape2:{}'.format(img_feats[2].shape))
            print("1------------------------------------------------------")
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
                print("2------------------------------------------------------")
                print('neck.shape0:{}'.format(img_feats[0].shape))
                print('neck.shape1:{}'.format(img_feats[1].shape))
                print('neck.shape2:{}'.format(img_feats[2].shape))
                print("2------------------------------------------------------")
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


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

#mp,新代码:


#mp,frj:

#from mmyolo.models.data_preprocessors import YOLOv5DetDataPreprocessor
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


