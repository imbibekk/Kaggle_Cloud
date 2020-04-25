import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import albumentations as A
from albumentations.pytorch import ToTensor
from torchvision.utils import make_grid

HEIGHT, WIDTH = 350, 525 
PAD_HEIGHT, PAD_WIDTH = 352, 544

def rle2mask(height, width, encoded):
    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T


class CloudDataset(Dataset):
    def __init__(self, df, data_dir, mode ='train', transform=None):

        self.df = df #.set_index('Image')
        self.image_ids = list(self.df.Image.unique())
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)
    
    def make_mask(self, df, image_id, image):
        H,W,C =  image.shape
        mask = np.zeros((H,W,4), dtype=np.uint8)
        encoded_masks = df.loc[df['Image'] == image_id, 'EncodedPixels']
        for idx, encoded in enumerate(encoded_masks.values):
            if encoded is not np.nan:
                mask[:,:,idx] = rle2mask(H, W, encoded)    
        return mask


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_dir = os.path.join(self.data_dir, image_id)
        image = cv2.imread(image_dir)
        if self.mode == 'test':
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            data = {}
            data['image'] = torch.tensor(image)
            data['image_id'] = image_id
            return data
        else:
            mask = self.make_mask(self.df, image_id, image)
            if self.transform is not None:
                augmented = self.transform(image=image, mask= mask)
                image = augmented['image']
                mask = augmented['mask']
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

            image = torch.tensor(image)
            mask = torch.tensor(mask)
    
            data = {}
            data['image'] = image
            data['mask'] = mask
            data['image_id'] = image_id
            return data


class PreTraining(Dataset):
    def __init__(self, data_dir, transform=None):
        train_imgs = glob.glob(os.path.join(data_dir, 'train_images')+'/*')
        test_imgs = glob.glob(os.path.join(data_dir, 'test_images')+'/*')
        images = []
        images.extend(train_imgs)
        images.extend(test_imgs)
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        image = cv2.imread(image_file)
        
        mask = cv2.imread(image_file, 0)
        mask = (mask>115).astype(int)
        mask = np.stack([mask, mask, mask, mask]).transpose(1,2,0)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask= mask)
            image = augmented['image']
            mask = augmented['mask']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.tensor(image)
        mask = torch.tensor(mask)
    
        data = {}
        data['image'] = image
        data['mask'] = mask
        data['image_id'] = os.path.basename(image_file)
        return data


def generate_transforms():
    
    train_transform = A.Compose([
                                A.Resize(HEIGHT, WIDTH),
                                A.PadIfNeeded(PAD_HEIGHT, PAD_WIDTH, border_mode=0),
                                A.ShiftScaleRotate(0.5, 0, 0, border_mode=0),
                                A.HorizontalFlip(),
                                A.VerticalFlip(),
                                A.OneOf([
                                    A.IAASharpen(alpha=(0.1, 0.3), p=0.5),
                                    A.CLAHE(p=0.8),
                                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                                ], p=0.8),
                                A.RandomBrightnessContrast(p=0.8),
                                A.RandomGamma(p=0.8),
                                A.Normalize(),
                                ])
    
    valid_transform = A.Compose([
                                A.Resize(HEIGHT, WIDTH),
                                A.PadIfNeeded(PAD_HEIGHT, PAD_WIDTH, border_mode=0),
                                A.Normalize(),
                                ])
    
    
    return train_transform, valid_transform


def get_dataloader(data_dir, df, mode='train', pretrain=False, batch_size=64):
    
    train_transform, valid_transform = generate_transforms()

    if pretrain:
        datasets = PreTraining(data_dir, transform=train_transform if mode =='train' else valid_transform)
    else:
        datasets = CloudDataset(df, data_dir, mode, transform=train_transform if mode =='train' else valid_transform)
    
    is_train = mode =='train'
    dataloader = DataLoader(datasets,
                            batch_size=batch_size if is_train else 2*batch_size, 
                            num_workers=4,
                            )
    return dataloader

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_dir = '../../data/train_images'
    mode = 'valid'
    fold = 0
    df = pd.read_csv('../../data/train_splits.csv')
    print(df.head())
    df.fillna('')
    df_train = df[df['Fold']!=fold]

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()
    
    loader = get_dataloader(data_dir, df_train, mode, batch_size=16)
    for data in loader:
        images = data['image']
        masks = data['mask']
        #print(masks.shape)
        show(make_grid(images))
        #show(make_grid(masks))
        break
