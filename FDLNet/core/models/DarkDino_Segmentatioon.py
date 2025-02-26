# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from . import box_ops
from ..utils.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class UpsamplingLayer(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, upscale = 2):
        super().__init__()
        self.dim = dim
        
        self.bilinear_upsample = nn.UpsamplingBilinear2d(scale_factor=upscale)  # 2 times upsampling
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=3, padding=1)
        self.att = CBAM(dim)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        ''' x: B C H W '''
        #x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.bilinear_upsample(x)  # (b c oh ow)
        x = self.att(x) # (b oc oh ow)
        x = self.conv(x)  # (b oc oh ow)
        x = self.norm(x)
        #x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x

def reshape_and_pad_tensor(tensor, target_shape=(2, 256, 64, 128)):
    """
    将形状为 [1, 256, 900] 的张量转换为目标形状 [1, 256, 64, 128]，如果有空余部分用0填充。
    
    参数:
    tensor (torch.Tensor): 输入的张量，形状为 [1, 256, 900]。
    target_shape (tuple): 目标形状，默认为 (1, 256, 64, 128)。
    
    返回:
    torch.Tensor: 形状为 target_shape 的张量。
    """
    original_shape = tensor.shape
    # if original_shape != (1, 256, 900):
    #     raise ValueError(f"输入张量的形状应为 (1, 256, 900)，但得到的是 {original_shape}")

    # 创建一个新的张量，初始化为0，并将原始张量的数据复制过来
    reshaped_tensor = torch.zeros(target_shape)
    
    # 计算原始张量的总元素数
    original_numel = tensor.numel()
    
    # 将原始张量的数据复制到新张量中
    reshaped_tensor.view(-1)[:original_numel] = tensor.view(-1)
    
    return reshaped_tensor

class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = 256, 8 #detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [768, 384, 192], hidden_dim)
        self.up1 = UpsamplingLayer(256,192)
        self.up2 = UpsamplingLayer(192+256,128)
        self.up3 = UpsamplingLayer(128+256, 64)
        self.up4 = UpsamplingLayer(64+256+256, 256)
        self.final = UpsamplingLayer(256, 20, upscale=4)
        #self.up5 = UpsamplingLayer(256,128)
        #self.up6 = UpsamplingLayer(128, 64)

    def forward(self, samples: NestedTensor):
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)
        # features, pos = self.detr.backbone(samples)

        # bs = features[-1].tensors.shape[0]

        # src, mask = features[-1].decompose()
        # assert mask is not None
        # src_proj = self.detr.input_proj(src)
        # hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        

        # outputs_class = self.detr.class_embed(hs)
        # outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        
        out, mask, hs, reference, src_proj,  features , memory = self.detr(samples, ['Night cars and objects.'] * 2)
        bs = features[-1].tensors.shape[0]

        hs_out = hs[-1].permute(0,2,1).contiguous()
        #print(hs_out.shape)

        hs_out = reshape_and_pad_tensor(hs_out)

        #print(features.tensors.shape)
        #print(memory.shape)
        
        layer1 = memory[:, :, :64*128].view(bs,256,64,128).to('cuda')
        layer2 = memory[:, :, 64*128: 64*128+ 32*64].view(bs,256,32,64).to('cuda')
        layer3 = memory[:, :, 64*128+32*64: 64*128+32*64+ 16*32].view(bs,256,16,32).to('cuda')
        layer4 = memory[:, :, 64*128+32*64+16*32:64*128+32*64+16*32 + 8*16].view(bs,256,8,16).to('cuda')
      
        #print(layer1.shape,layer2.shape,layer3.shape,layer4.shape)
        #memory = layer1
        #print('memory', memory.shape)
        #hs_out = torch.cat((layer1.to('cuda'),hs_out.to('cuda')), dim = 1) # 1,512,64,128
        hs_out = hs_out.to('cuda')
        x = layer4.to('cuda')
        x = self.up1(x) # 256 -> 192
        x = torch.cat((x,layer3), dim = 1) # 192 + 256
        x = self.up2(x) # 192+256 -> 128
        x = torch.cat((x, layer2), dim = 1) # 128 + 256
        x = self.up3(x) # 128 + 256 -> 64
        x = torch.cat((x,layer1,hs_out), dim = 1) # 256+256+64
        x = self.up4(x) # 256+256+64 -> 256
        x = self.final(x)
        # x = hs_out
        # x = self.up1(x)
        # x = self.up2(x)
        # x = self.up3(x)

        #print('x', x.shape)
        #print(mask.shape, memory.shape, src_proj.shape)
        #memory = memory.permute(1,2,0).view(bs,256,,w)
        #out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        # if self.detr.aux_loss:
        #     out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)
        
        # FIXME h_boxes takes the last one computed, keep this in mind
        # bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        # seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        # out["pred_masks"] = outputs_seg_masks
        return x



def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
    
#zmp:
# class YOLOWDETRsegm(nn.Module):
#     def __init__(self, yolow):
#         """
#         Args:
#             class_texts: 分割类别文本描述列表，如['car', 'person', ...]
#         """
#         super().__init__()
#         self.yolow = yolow
#         class_texts=["Void",
#                      "Road",
#                      "Sidewalk",
#                      "Building",
#                      "Wall",
#                      "Fence",
#                      "Pole",
#                      "Traffic Light",
#                      "Traffic Sign",
#                      "Vegetation",
#                      "Terrain",
#                      "Sky",
#                      "People",
#                      "Rider",
#                      "Car",
#                      "Truck",
#                      "Bus",
#                      "Train",
#                      "Motorcycle",
#                      "Bicycle"]
#          # 预注册文本特征
#         self.yolow.reparameterize([class_texts]*256)  # 假设batch_size=256需预先扩展
        
#         hidden_dim, nheads = 256, 8 #detr.transformer.d_model, detr.transformer.nhead
#         self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
#         self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [768, 384, 192], hidden_dim)
#         self.up1 = UpsamplingLayer(256,192)
#         self.up2 = UpsamplingLayer(192+256,128)
#         self.up3 = UpsamplingLayer(128+256, 64)
#         self.up4 = UpsamplingLayer(64+256+256, 256)
#         self.final = UpsamplingLayer(256, 20, upscale=4)
    #     # 通道适配模块（匹配YOLO-World输出）
    #     self.channel_adapters = nn.ModuleList([
    #         nn.Sequential(
    #             nn.Conv2d(320, 256, 1),  # p3/160
    #             nn.GroupNorm(8, 256)
    #         ),
    #         nn.Sequential(
    #             nn.Conv2d(640, 256, 1),  # p4/80
    #             nn.GroupNorm(16, 256)
    #         ),
    #         nn.Sequential(
    #             nn.Conv2d(640, 256, 1),  # p5/40
    #             nn.GroupNorm(16, 256)
    #         )
    #     ])
        
    #     # 特征金字塔构建模块
    #     self.fpn = nn.ModuleList([
    #         self._build_up_block(256, 192),       # 40->80
    #         self._build_up_block(192+256, 128),   # 80->160 
    #         self._build_up_block(128+256, 64),    # 160->320
    #         self._build_up_block(64, 20)          # 最终输出
    #     ])

    # def forward(self, samples: NestedTensor):
    #     img_feats = self.yolow(
    #         batch_inputs=samples.tensors,  # 从NestedTensor中提取张量
    #         batch_data_samples={'texts': self.yolow.texts}
    #     )
        
    #     # Step 2: 通道适配（保持与DETRsegm兼容）
    #     p3 = self.channel_adapters[0](img_feats[0])  # [B,256,160,160]
    #     p4 = self.channel_adapters[1](img_feats[1])  # [B,256,80,80] 
    #     p5 = self.channel_adapters[2](img_feats[2])  # [B,256,40,40]

    #     # Step 3: 特征金字塔构建（自顶向下）
    #     x = self.fpn[0](p5)              # 40->80 [B,192,80,80]
    #     x = torch.cat([x, p4], dim=1)    # 拼接中层特征 [B,192+256=448,80,80]
    #     x = self.fpn[1](x)               # 80->160 [B,128,160,160]
        
    #     x = torch.cat([x, p3], dim=1)    # 拼接浅层特征 [B,128+256=384,160,160]
    #     x = self.fpn[2](x)               # 160->320 [B,64,320,320]
        
    #     # Step 4: 分割预测
    #     masks = self.fpn[3](x)            # 320->640 [B,20,640,640]
    #     return masks


#frj

class YOLOWDETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr
        # 保持原有的上采样模块
        self.up1 = UpsamplingLayer(512, 192)
        self.up2 = UpsamplingLayer(192 + 512, 128)
        self.up3 = UpsamplingLayer(128 + 256, 64)
        self.final = UpsamplingLayer(64, 19, upscale=4)
        #frj
        self.up_test1=UpsamplingLayer(2560, 192)
        self.up_test2=UpsamplingLayer(192+1280, 128)
        self.up_test3=UpsamplingLayer(128+640, 64)
        self.final_test=UpsamplingLayer(64, 19, upscale=4)
    def forward(self, samples: NestedTensor):
        # 仅获取 features（假设返回的是多尺度特征列表）
        # class_texts=[
        #      'Void',
        #      'Road',
        #      'Sidewalk',
        #      'Building',
        #      'Wall',
        #      'Fence',E
        #      'Pole',
        #      'Traffic Light',
        #      'Traffic Sign',
        #      'Vegetation',
        #      'Terrain',
        #      'Sky',
        #      'People',
        #      'Rider',
        #      'Car',
        #      'Truck',
        #      'Bus',
        #      'Train',
        #      'Motorcycle',
        #      'Bicycle']
        # class_texts=1
        class_texts=[
                     "Road",
                     "Sidewalk",
                     "Building",
                     "Wall",
                     "Fence",
                     "Pole",
                     "Traffic Light",
                     "Traffic Sign",
                     "Vegetation",
                     "Terrain",
                     "Sky",
                     "People",
                     "Rider",
                     "Car",
                     "Truck",
                     "Bus",
                     "Train",
                     "Motorcycle",
                     "Bicycle"
                     ]
         # 预注册文本特征
        
        self.detr.reparameterize([class_texts]*(samples.shape[0]))  # 假设batch_size=256需预先扩展
        features,_ = self.detr.extract_feat(samples,None)  # 返回格式：features = [p3, p4, p5]
        # print(type(features))
        # print('features0 shape:{}'.format(features[0].shape))
        # print('features1 shape:{}'.format(features[1].shape))
        # print('features2 shape:{}'.format(features[2].shape))
        # 明确层级特征（假设 features[0] 是最底层特征）
        layer1 = features[0]  # [B, C, H1, W1] 例如 [B, 256, 64, 128]
        layer2 = features[1]  # [B, C, H2, W2] 例如 [B, 256, 32, 64]
        layer3 = features[2]  # [B, C, H3, W3] 例如 [B, 256, 16, 32]

        # 特征金字塔上采样流程
        x = layer3
        # x = self.up1(x)  # [B, 192, H3, W3]
        # x = torch.cat([x, layer2], dim=1)  # 拼接中层特征
        # x = self.up2(x)  # [B, 128, H2, W2]
        # x = torch.cat([x, layer1], dim=1)  # 拼接中层特征
        # x = self.up3(x)  # [B, 64, H1, W1]

        x = self.up_test1(x)  # [B, 192, H3, W3]
        x = torch.cat([x, layer2], dim=1)  # 拼接中层特征
        x = self.up_test2(x)  # [B, 128, H2, W2]
        x = torch.cat([x, layer1], dim=1)  # 拼接中层特征
        x = self.up_test3(x)  # [B, 64, H1, W1]
        # 最终上采样到原图尺寸
        masks = self.final_test(x)  # [B, 20, H_orig, W_orig]
        print("mask_shape:{}".format(masks.shape))
        return masks
    
