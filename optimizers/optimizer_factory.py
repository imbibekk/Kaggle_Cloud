from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, Conv2d, Linear

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                    amsgrad=amsgrad)


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
  return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                   nesterov=nesterov)

def adamW(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
      return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
      

def get_optimizer(args, model):

    # build optimizer
    if args.no_bias_decay: 
        if args.encoder_lr_ratio:
            encoder_lr_ratio = args.encoder_lr_ratio
            group_decay_encoder, group_no_decay_encoder = group_weight(model.model.encoder)
            group_decay_decoder, group_no_decay_decoder = group_weight(model.model.decoder)
            base_lr = args.lr
            params = [{'params': group_decay_decoder},
                      {'params': group_no_decay_decoder, 'weight_decay': 0.0},
                      {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                      {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
        else:
            group_decay, group_no_decay = group_weight(model)
            params = [{'params': group_decay},
                      {'params': group_no_decay, 'weight_decay': 0.0}]
    else:
        params = model.parameters()
    
    if args.optimizer_name == 'sgd':
        return sgd(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer_name == 'adam':
        return adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer_name == 'adamW':
        return adamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
