"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']


# TODO: optim function
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    #frj
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):

        *preds, target = tuple(inputs)
        #frj
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)

class FDLNetLoss(nn.Module):
    def __init__(self, classes=19, ignore_index=255, norm=False, upper_bound=1.0, mode='train', 
                aux_weight=0.4, seg_weight=1, att_weight=0.01, **kwargs):
        super(FDLNetLoss, self).__init__()
        self.num_classes = classes
        self.ignore_index = ignore_index
        if mode == 'train':
            self.seg_loss = OhemCrossEntropy2d(
                                               ignore_index=ignore_index)
        elif mode == 'val':
            self.seg_loss = OhemCrossEntropy2d(
                                               ignore_index=ignore_index)

        self.aux_loss = OhemCrossEntropy2d(ignore_index=ignore_index)
        self.att_loss = OhemCrossEntropy2d(min_kept=5000,ignore_index=ignore_index)
        self.aux_weight = aux_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight


    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * self.ignore_index
        return self.att_loss(input, 
                             torch.where(edge.max(1)[0] == 1., target, filler))

    # def forward(self, inputs, targets):
    #     pred1, pred2 = inputs
    #     # print(pred1.shape, pred2.shape)
    #     target, edgemap =targets

    #     losses = {}

    #     losses['seg_loss'] = self.seg_weight * self.seg_loss(pred1, target)
    #     losses['aux_loss'] = self.aux_weight * self.aux_loss(pred2, target)

    #     losses['att_loss'] = self.att_weight * self.edge_attention(pred1, target, edgemap)
        
    #     return losses
    #frj
    def forward(self, inputs, targets):
        pred1= inputs
        # print(pred1.shape, pred2.shape)
        target=targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(pred1, target)
        losses['aux_loss'] = 0

        losses['att_loss'] = 0
        
        return losses

class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))
#frj
class SegmentationCrossEntropyLoss(nn.Module):
    def __init__(self, aux=False,aux_weight=0.2,ignore_index=-100, class_weights=[0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507]):
        """
        初始化参数:
        :param ignore_index: 需要忽略的掩码标签索引（默认-100）
        :param class_weights: 各类别的权重张量（Tensor），用于处理类别不平衡问题
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        前向计算:
        :param inputs: 模型预测输出 [batch_size, num_class, h, w]
        :param targets: 分割掩码标签 [batch_size, h, w]
        :return: 计算得到的损失值
        """
        # 确保输入形状正确
        assert inputs.dim() == 4, f"Input must be 4D tensor, got {inputs.dim()}D"
        assert targets.dim() == 3, f"Target must be 3D tensor, got {targets.dim()}D"

        # 转换为交叉熵损失要求的形状: (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.size(1))
        
        # 目标形状转换: (N, H, W) -> (N*H*W,)
        targets_flat = targets.view(-1)

        # 创建mask排除需要忽略的像素
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            inputs_flat = inputs_flat[mask]
            targets_flat = targets_flat[mask]

        # 如果没有有效像素，返回0损失
        if targets_flat.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)

        # 计算交叉熵损失
        return F.cross_entropy(
            inputs_flat,
            targets_flat,
            weight=self.class_weights.to(inputs.device) if self.class_weights is not None else None,
            reduction='mean'
        )

def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)

    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'fdlnet':
        return FDLNetLoss(**kwargs)
    elif model=='yoloworld':
        #return SegmentationCrossEntropyLoss(**kwargs)
        return FDLNetLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
