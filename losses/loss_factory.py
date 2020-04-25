from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)

    if input.dim() == 4:
        B,C,H,W = input.size()
        iflat = input.view(B*C,-1)
        tflat = target.view(B*C,-1)
    else:
        assert input.dim() == 3
        B,H,W = input.size()
        iflat = input.view(B,-1)
        tflat = target.view(B,-1)
    intersection = (iflat * tflat).sum(dim=1)
                
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth))
    loss = loss.mean()
    return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, target):
        return dice_loss(input, target)


def segmentation_loss(logits, targets):
    loss_bce = nn.BCEWithLogitsLoss() 
    loss_dice = DiceLoss()
    loss_1 = loss_bce(logits, targets)
    loss_2 = loss_dice(logits, targets)
    return 0.2*loss_1 + 0.8*loss_2


def pre_training(logits, targets):
    loss_bce = nn.BCEWithLogitsLoss() 
    loss_dice = DiceLoss()
    loss_1 = loss_bce(logits, targets)
    loss_2 = loss_dice(logits, targets)
    return 0.2 * loss_1 + 0.8 * loss_2


def stage2_loss(logits_maks, logits_cls, targets_mask, target_cls):
    seg_loss = segmentation_loss(logits_maks, targets_mask)
    loss_bce = nn.BCEWithLogitsLoss() 
    cls_loss = loss_bce(logits_cls, target_cls)
    return 0.2* cls_loss + 0.8 * seg_loss



def get_loss(args):
    if args.pretrain:
        return pre_training
    else:
        return stage2_loss
    