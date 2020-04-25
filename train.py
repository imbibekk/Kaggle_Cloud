from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import datetime
import pprint
import tqdm
import copy

import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
import torch.nn.functional as F

from datasets import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
from utils import checkpoint
from utils.metrics import compute_metrics
from utils.utils import Logger, seed_everything, update_avg


def evaluate_single_epoch(args, model, dataloader, criterion):

    model.eval()
    curr_loss_avg = 0
    valid_preds, cls_preds, valid_targets = [], [], [] 
    
    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))
    with torch.no_grad():
        for batch_idx, data in enumerate(tbar):
            images = data['image'].cuda()
            masks = data['mask'].cuda()

            masks_logits, class_logits = model(images)  
            masks_prob = torch.sigmoid(masks_logits)
            cls_probs = torch.sigmoid(class_logits)

            valid_preds.append(masks_prob.detach().cpu())
            cls_preds.append(cls_probs.detach().cpu())
            valid_targets.append(masks.detach().cpu())

            B,C,H,W = masks.size()
            cls_labels = masks.view(B, C, H*W)
            cls_labels = torch.sum(cls_labels, dim=2)
            cls_labels = (cls_labels > 0).float()
            
            if args.pretrain:
                loss = criterion(masks_logits, masks)  
            else:
                loss = criterion(masks_logits, class_logits, masks, cls_labels)

            curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        
            tbar.set_description('loss: {:.4}'.format(curr_loss_avg.item()))

        valid_preds = torch.cat(valid_preds)
        cls_preds = torch.cat(cls_preds)
        valid_targets = torch.cat(valid_targets)
  
        B,C,H,W = valid_preds.size()
        if 350 != H and 525 != W:
            valid_preds = F.interpolate(valid_preds, size=(350,525),mode='bilinear')
            valid_targets = F.interpolate(valid_targets, size=(350,525),mode='bilinear')

        # convert to numpy
        valid_preds = valid_preds.numpy()
        cls_preds = cls_preds.numpy()
        valid_targets = valid_targets.numpy()

        assert valid_preds.shape == valid_targets.shape
        cls_thres = np.array([0.7,0.7,0.7,0.7])
        thres = np.array([0.4,0.4,0.4,0.4])
        
        predictions = (valid_preds > thres[None,:,None,None]).astype(int)
        cls_predictions = (cls_preds > cls_thres).astype(int)
        predictions = predictions * cls_predictions[:,:,None,None]   
        B,C,H,W = predictions.shape
        mean_dice, non_empty_dice, cls_accuracy, precision, recall, _, _ = compute_metrics(predictions.reshape(-1,H,W), valid_targets.reshape(-1,H,W))
        
    return curr_loss_avg.item(), mean_dice, non_empty_dice, cls_accuracy, precision, recall


def train_single_epoch(args, model, dataloader, criterion, optimizer, scheduler):
    
    model.train()
    curr_loss_avg = 0
    tbar = tqdm.tqdm(dataloader,  total=len(dataloader))
    for batch_idx, data in enumerate(tbar):
        images = data['image'].cuda()
        masks = data['mask'].cuda()

        masks_logits, class_logits = model(images)  
        
        B,C,H,W = masks.size()
        cls_labels = masks.view(B, C, H*W)
        cls_labels = torch.sum(cls_labels, dim=2)
        cls_labels = (cls_labels > 0).float()
            
        if args.pretrain:
            loss = criterion(masks_logits, masks)  
        else:
            loss = criterion(masks_logits, class_logits, masks, cls_labels)
        
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if args.scheduler_name in ['linear_warmup', 'onecycle', 'cyclic']:
            scheduler.step()

        curr_loss_avg = update_avg(curr_loss_avg, loss, batch_idx)        
        tbar.set_description('loss: %.5f, lr: %.6f' % (curr_loss_avg.item(), optimizer.param_groups[0]['lr']))
    return curr_loss_avg.item()

        
def train(args, log, model, dataloaders, criterion, optimizer, scheduler, start_epoch):
    num_epochs = args.num_epochs

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    log.write('Start Training...!!\n')

    patience = 0.0
    best_val_loss = 10.0
    best_dice_score = 0.0
    best_train_loss = 10.0
    for epoch in range(start_epoch, num_epochs):

        # train phase
        train_loss = train_single_epoch(args, model, dataloaders['train'], criterion, optimizer, scheduler)
        
        if args.pretrain:
            log_write = f'Epoch: {epoch} | Train_loss: {train_loss:.5f}'
            log.write(log_write)
            log.write('\n')

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                name = f'train_loss'
                checkpoint.save_checkpoint(args, model, optimizer, epoch, step=0, keep=args.ckpt_keep, name=name)
        else:
            # valid phase
            val_loss, mean_dices, non_empty_dices, cls_accuracies, precisions, recalls = evaluate_single_epoch(args, model, dataloaders['val'], criterion)
    
            log_write = f'Epoch: {epoch} | Train_loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | mean dice: {mean_dices:.5f} | non_empty_dices: {non_empty_dices:.5f} | cls_accuracies: {cls_accuracies:.5f} | precisions: {precisions:.5f} | recalls: {recalls:.5f}'
            log.write(log_write)
            log.write('\n')
            
            if args.scheduler_name == 'multistep':
                scheduler.step()
        
            # save metric checkpoint
            name = 'metric' 
            checkpoint.save_checkpoint(args, model, optimizer, epoch=epoch, metric_score=mean_dices, step=0, keep=args.ckpt_keep, name=name)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
            if mean_dices > best_dice_score:
                patience = 0.0
                best_dice_score = mean_dices
            else:
                patience += 1
                if patience ==args.patience:
                    log.write(f'Early Stopping....@ {epoch} epoch for patience @ {patience}\n')
                    log.write(f'Best Loss: {best_val_loss} | Best Dice: {best_dice_score}\n')        
                    break
        
            log.write(f'Best Loss: {best_val_loss} | Best Dice: {best_dice_score}\n')

        
def run(args, log):
    
    df = pd.read_csv(args.df_path)
    df_train = df[df['fold']!=args.fold]
    df_valid = df[df['fold']==args.fold]
    dfs = {}
    dfs['train'] = df_train
    dfs['val'] = df_valid
    
    model = get_model(args).cuda()

    if args.stage == 'stage1':  # segmentation only training; freeze classification head
        log.write(f'Freezing classification head for {args.stage}\n')
        for param in model.model.classification_head.parameters():
            param.requires_grad = False   
    
    elif args.stage == 'stage2': # classification only training; freeze segmentation head
        log.write(f'Training only classification head for {args.stage}\n')
        for param in model.model.encoder.parameters():
            param.requires_grad = False

        for param in model.model.decoder.parameters():
            param.requires_grad = False

        for param in model.model.segmentation_head.parameters():
            param.requires_grad = False   
    
    criterion = get_loss(args)
    optimizer = get_optimizer(args, model)

    if args.initial_ckpt is not None:
        last_epoch, step = checkpoint.load_checkpoint(args, model, checkpoint=args.initial_ckpt)
        log.write(f'Resume training from {args.initial_ckpt} @ {last_epoch}\n')
    last_epoch, step = -1, -1

    dataloaders = {mode:get_dataloader(args.data_dir, dfs[mode], mode, args.pretrain, args.batch_size) for mode in ['train', 'val']}   
    scheduler = get_scheduler(args, optimizer, -1, dataloaders['train'])
    seed_everything()

    train(args, log, model, dataloaders, criterion, optimizer, scheduler, last_epoch+1)


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
    parser.add_argument('--pretrain',type=bool, default=False,
                    help='whether pretrain the network using train+test images')
    parser.add_argument('--stage',type=str, default=None,
                    help='stage1 or stage2 for training')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')                 
    parser.add_argument('--batch_size', type=int, default=8, 
                    help='batch size')
    parser.add_argument('--num_epochs', type=int, default=5, 
                    help='num of epochs to train')
    parser.add_argument('--grad_clip', type=float, default=None,
                    help='clipping value for gradient')
    parser.add_argument('--optimizer_name', type=str, default='adamW',
                    help='name of optimizer to use')
    parser.add_argument('--scheduler_name', type=str, default='multistep',
                    help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate for scheduler')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1,
                    help='relative learning rate-ratio for encode')
    parser.add_argument('--no_bias_decay', type=bool, default=True,
                    help='wheter to apply weight decay to bias or not?')   
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay for optimizer')   
    parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stopping')   
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='FPN',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=4,
                    help='number of classes')                 
    parser.add_argument('--initial_ckpt', type=str, default=None,
                    help='inital checkpoint to resume training')
    parser.add_argument('--ckpt_keep', type=int, default=5,
                    help='how many checkpoints to save')                              
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
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/train_log.txt', mode='a')
    log.write('start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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


