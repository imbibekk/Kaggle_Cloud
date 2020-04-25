from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
import segmentation_models_pytorch as smp


class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        logits_mask,logits_class = [], []
        with torch.no_grad():
            for m in self.models:
                logits_m, logits_c = m(x)
                logits_mask.append(logits_m)
                logits_class.append(logits_c)

        logits_mask = torch.stack(logits_mask)
        logits_mask = torch.mean(logits_mask, dim=0)
        
        logits_class = torch.stack(logits_class)
        logits_class = torch.mean(logits_class, dim=0)

        return logits_mask, logits_class


class CloudModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, num_class):
        super(CloudModel, self).__init__()
        self.aux_params=dict(
                    pooling='avg',              # one of 'avg', 'max'
                    dropout=None,               # dropout ratio, default is None
                    activation=None,            # activation function, default is None
                    classes=num_class,          # define number of output labels
                    )
        if decoder_name == 'Unet':
            self.model = smp.Unet(encoder_name, classes=num_class, aux_params=self.aux_params)
        elif decoder_name == 'FPN':
            self.model = smp.FPN(encoder_name, classes=num_class, aux_params=self.aux_params)
        else:
            raise NotImplemented

    def forward(self, x):
        outmask, label = self.model(x)
        return outmask, label


def get_model(args):
    return CloudModel(encoder_name=args.encoder_name, decoder_name=args.decoder_name, num_class=args.num_class)


if __name__ == '__main__':
   
    import argparse
    
    parser = argparse.ArgumentParser(description='Kaggle Cloud Competition')
    parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='Unet', 
                    help='which decoder to use for model')
    parser.add_argument('--num_class', type=int, default=4,
                    help='number of classes')               
    args = parser.parse_args()

    model = get_model(args)
    print(model)
    
    dump_inp = torch.randn((16, 3, 512, 512))
    logits_mask, logits_class = model(dump_inp)
    print(logits_mask.shape, logits_class.shape)