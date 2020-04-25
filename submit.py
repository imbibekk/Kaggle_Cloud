from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pandas as pd
import numpy as np
import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataloader
from models import get_model
from utils import checkpoint
from utils.utils import Logger, seed_everything


LABEL_LIST = ['Fish', 'Flower', 'Gravel', 'Sugar']


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(height, width, encoded):
    if isinstance(encoded, float):
        img = np.zeros((height,width), dtype=np.uint8)
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


def logit_to_probability(logit_mask, logit_label):
    probability_mask  = torch.sigmoid(logit_mask )
    probability_label = torch.sigmoid(logit_label)
    return probability_mask, probability_label


def inference_submit(model, dataloader, augment):
    
    test_ids = [] 
    test_mask_predictions  = [] 

    cls_thres = np.array([0.7,0.7,0.7,0.7])
    thres = np.array([0.4,0.4,0.4,0.4])

    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            images = data['image'].cuda()
            image_id = data['image_id']
            
            num_augment=0
            probability_mask=0
            probability_label=0
            
            if 'null' in augment: #1: #  null
                logit_mask, logit_label = model(images)  #net(input)
                p_mask,  p_label = logit_to_probability(logit_mask, logit_label)

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            if 'flip_lr' in augment:
                logit_mask, logit_label = model(torch.flip(images,dims=[3]))
                p_mask, p_label = logit_to_probability(torch.flip(logit_mask,dims=[3]), logit_label)

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            if 'flip_ud' in augment:
                logit_mask, logit_label = model(torch.flip(images,dims=[2]))
                p_mask, p_label = logit_to_probability(torch.flip(logit_mask,dims=[2]), logit_label)

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            if 'flip_both' in augment:
                logit_mask, logit_label = model(torch.flip(images,dims=[2,3]))
                p_mask, p_label = logit_to_probability(torch.flip(logit_mask,dims=[2,3]), logit_label)

                probability_mask  += p_mask
                probability_label += p_label
                num_augment+=1

            probability_mask  = probability_mask/num_augment
            probability_label = probability_label/num_augment
            
            B,C,H,W = probability_mask.size()
            if 350 != H and 525 != W:
                probability_mask = F.interpolate(probability_mask, size=(350,525),mode='bilinear')

            probability_mask = probability_mask.detach().cpu().numpy()
            probability_label = probability_label.detach().cpu().numpy()

            mask_predictions = (probability_mask > thres[None,:,None,None]).astype(int)
            cls_predictions = (probability_label > cls_thres).astype(int)
            mask_predictions = mask_predictions * cls_predictions[:,:,None,None] 

            test_ids.extend(image_id)
            test_mask_predictions.append(mask_predictions)

        test_mask_predictions = np.concatenate(test_mask_predictions)
        return test_ids, test_mask_predictions



def submit(args, log):
    df = pd.read_csv(args.df_path)
    df['Image'] = df.Image_Label.map(lambda v: v[:v.find('_')])
    print(df.head())

    model = get_model(args).cuda()
    last_epoch, step  = checkpoint.load_checkpoint(args, model, checkpoint=args.initial_ckpt)
    log.write(f'Loaded checkpoint from {args.initial_ckpt} @ {last_epoch}\n')

    dataloader = get_dataloader(args.data_dir, df, 'test', args.pretrain, args.batch_size) 
    seed_everything()

    # inference
    test_ids, mask_predictions = inference_submit(model, dataloader, args.tta_augment)

    assert len(test_ids) == mask_predictions.shape[0]
    
    ids = []
    rles = []
    for i, image_id in tqdm.tqdm(enumerate(test_ids), total=len(test_ids)):    
        predictions = mask_predictions[i]
        for cls_idx in range(4):
            prediction = predictions[cls_idx,:,:]
            H,W = prediction.shape
            assert H == 350 and W == 525
            rle_encoded = mask2rle(prediction)
            assert np.all(rle2mask(H, W, rle_encoded) == prediction)
            ids.append(f'{image_id}_{LABEL_LIST[cls_idx]}')
            rles.append(rle_encoded)
    
    df_submission = pd.DataFrame({'Image_Label': ids, 'EncodedPixels': rles})
    df_submission.to_csv(args.sub_name, index=False)
    print(df_submission.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Cloud Competition')
    parser.add_argument('--gpu', type=int, default=0, 
                    help='Choose GPU to use. This only support single GPU')
    parser.add_argument('--data_dir', default='./data/test_images',
                    help='datasest directory')
    parser.add_argument('--df_path', default='./data/sample_submission.csv',
                    help='df_path')
    parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for training')
    parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size')
    parser.add_argument('--model_name', type=str, default='FPN_effb4',
                    help='model_name as exp_name')
    parser.add_argument('--pretrain', type=bool, default=False,
                    help='number of classes')                
    parser.add_argument('--encoder_name', type=str, default='efficientnet-b4', 
                    help='which encode to use for model')
    parser.add_argument('--decoder_name', type=str, default='FPN',
                    help='type of decoder to use for model')
    parser.add_argument('--num_class', type=int, default=4,
                    help='number of classes')
    parser.add_argument('--initial_ckpt', type=str, default=None,
                    help='inital checkpoint to resume training')
    parser.add_argument('--tta_augment', type=list, default=['null', 'flip_lr', 'flip_ud', 'flip_both'], 
                    help='logging directory')                
    parser.add_argument('--sub_name', type=str, default='submission.csv',
                    help='name for submission df')
    parser.add_argument('--log_dir', type=str, default='runs', 
                    help='logging directory')                

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= f'{args.gpu}'

    log = Logger()
    #log.open(args.log_dir + '/submit_log.txt', mode='a')
    log.open(args.log_dir + '/' + args.model_name + f'/fold_{args.fold}' + '/submit_log.txt', mode='a')
    log.write('*'*30)
    log.write('\n')
    log.write('Logging arguments!!\n')
    log.write('*'*30)
    log.write('\n')
    for arg, value in sorted(vars(args).items()):
        log.write(f'{arg}: {value}\n')
    log.write('*'*30)
    log.write('\n')

    submit(args, log)
                     
    