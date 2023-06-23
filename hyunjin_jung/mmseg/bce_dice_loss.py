import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred*target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2*intersection + smooth)/(pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target.float())
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

@MODELS.register_module()
class BCEDiceLoss(nn.Module):
    def __init__(self, use_sigmoid, loss_weight):
        super(BCEDiceLoss, self).__init__()
        self._loss_name = "loss_bce_dice"
        self.reduction = 'mean'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = bce_dice_loss(pred, target)
        return loss
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name