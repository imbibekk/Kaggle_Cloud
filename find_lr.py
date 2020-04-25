from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import CLR
import utils
from utils import checkpoint
from utils.utils import Logger, seed_everything


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom


def train(args, model, dataloader, criterion, optimizer, clr):
    
    running_loss = 0.
    avg_beta = 0.98
    model.train()

    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))
    for batch_idx, data in enumerate(tbar):
    
        images = data['image'].cuda()
        labels = data['mask'].cuda()
    
        masks, cls_logits = model(images)
        
        B,C,H,W = labels.size()
        cls_labels = labels.view(B, C, H*W)
        cls_labels = torch.sum(cls_labels, dim=2)
        cls_labels = (cls_labels > 0).float()

        if args.mode == 'classification':
            loss = criterion(cls_logits, cls_labels)
        elif args.mode == 'segmentation':
            loss = criterion(masks, labels)
        elif args.pretrain:
            loss = criterion(masks, labels)
        else:
            loss = criterion(masks, cls_logits, labels, cls_labels)

        running_loss = avg_beta * running_loss + (1-avg_beta) *loss.data
        smoothed_loss = running_loss / (1 - avg_beta**(batch_idx+1))
        
        tbar.set_description('loss: %.5f' % (smoothed_loss))        
        
        lr = clr.calc_lr(smoothed_loss)
        if lr == -1 :
            clr.plot()
            plt.show()
            plt.savefig('lr_find.png')
            break
        update_lr(optimizer, lr)   
    
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run(args, log):

    df = pd.read_csv(args.df_path)
    df_train = df[df['Fold']!=args.fold]
    df_valid = df[df['Fold']==args.fold]
    dfs = {}
    dfs['train'] = df_train
    dfs['val'] = df_valid
    
    model = get_model(args).cuda()
    
    if args.mode != 'segmentation':
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        for param in model.model.decoder.parameters():
            param.requires_grad = True
        for params in model.model.classification_head.parameters():
            params.requires_grad = False

    elif args.mode == 'classification':
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        for param in model.classification_head.parameters():
            param.requires_grad = True    

    criterion = get_loss(args)
    optimizer = get_optimizer(args, model)
    
    if args.initial_ckpt is not None:
        last_epoch, step = checkpoint.load_checkpoint(args, model, checkpoint=args.initial_ckpt)
        log.write(f'Resume training from {args.initial_ckpt} @ {last_epoch}\n')
    else:
        last_epoch, step = -1, -1
    
    dataloaders = {mode:get_dataloader(args.data_dir, dfs[mode], mode, args.pretrain, args.batch_size) for mode in ['train', 'val']}   
    seed_everything(seed=123)
    clr = CLR(optimizer, len(dataloaders['train']))

    train(args, model, dataloaders['train'], criterion, optimizer, clr)


def parse_args():
    parser = argparse.ArgumentParser(description='Kaggle Cloud Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/train_images',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/train_splits.csv',
                    help='df_path')                 
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')                 
    parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size')
    parser.add_argument('--num_epochs', type=int, default=30, 
                    help='num of epochs to train')
    parser.add_argument('--mode', type=str, default=None, 
                    help='mode: segmentation or classification or both')
    parser.add_argument('--pretrain', type=bool, default=False,
                    help='number of classes')                
    parser.add_argument('--use_compressor', type=bool, default=False,
                    help='whether to use small network for extraction of features')
    parser.add_argument('--optimizer_name', type=str, default='adamW',
                    help='name of optimizer to use')
    parser.add_argument('--lr', type=float, default=0.0005,
                    help='minimum learning rate for scheduler')
    parser.add_argument('--max_lr', type=float, default=0.0001,
                    help='maximum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay for optimizer')   
    parser.add_argument('--no_bias_decay', type=bool, default=True,
                    help='wheter to apply weight decay to bias or not?')
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4', 
                    help='which encode to use for model')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                    help='relative learning rate-ratio for encode')
    parser.add_argument('--decoder_name', type=str, default='FPN',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=4,
                    help='number of classes')   
    parser.add_argument('--initial_ckpt', type=str, default=None,
                    help='inital checkpoint to resume training')                                        
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    return parser.parse_args()

def main():
    import warnings
    warnings.filterwarnings("ignore")
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'
    utils.prepare_train_directories(args)
    
    log = Logger()
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/findlr_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    run(args, log)
    print('success!')

if __name__ == '__main__':
    main()



