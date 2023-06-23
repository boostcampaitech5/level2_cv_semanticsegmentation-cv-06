# python native
import os
import json
import random
import datetime
from functools import partial
import torch.nn.functional as F


# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
from torch.utils.data import Dataset, DataLoader


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

# visualization
import matplotlib.pyplot as plt

from dataset import XRayDataset
from traintools import set_seed, dice_coef, save_model, validation, increment_path
from loss import *

import datetime
import pytz
import torch, gc
import argparse
import wandb
gc.collect()
torch.cuda.empty_cache()

from models import UNet

### 저장 폴더명 생성
kst = pytz.timezone('Asia/Seoul')
now = datetime.datetime.now(kst)
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S')

from PIL import Image
import PIL

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def train(args, aug):
    print(f'Start training..')

    set_seed(args.seed)

    SAVED_DIR = increment_path(os.path.join(args.model_dir, save_name))
    print(SAVED_DIR)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    valid_tf = A.Compose([A.Resize(512, 512)])
    train_tf = A.Compose([aug, A.Resize(512, 512)])
    train_dataset = XRayDataset(is_train=True, transforms=train_tf)
    valid_dataset = XRayDataset(is_train=False, transforms=valid_tf)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    # model = torch.load('./beit_base_patch16_640_pt22k_ft22ktoade20k.pth')
    model = models.segmentation.fcn_resnet101(pretrained=True)
    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    
    # Loss function 정의
    # criterion = nn.BCEWithLogitsLoss()
    criterion = bce_dice_loss
    
    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(args.epochs):
        model.train()

        for step, (images, masks) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            # inference
            outputs = model(images)['out']
            # outputs = model(images)
            
            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': loss})
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(), 4)}'
                )
        if not os.path.isdir(SAVED_DIR):
            os.makedirs(SAVED_DIR)

        if (epoch + 1) % 5 == 0:
            save_model(model, SAVED_DIR, file_name=f'{epoch+1}.pt')
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.interval == 0:
            dice = validation(epoch + 1, model, valid_loader, criterion)
            wandb.log({'val_dice': dice})
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model, SAVED_DIR)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='XRayDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='baseline', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=2, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='one-peace', help='model type')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='combined loss', help='criterion type (default: focal)')
    parser.add_argument('--interval', type=int, default=2, help='batch intervals for validation')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/changed_data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './trained_model'))
    args = parser.parse_args()

    for aug in [A.RandomCrop(2000, 2000, p=0.5), A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4)]:
        save_name = folder_name +  '-' + aug.__class__.__name__
        print(args)
        
        wandb.init(project='project3_Betty_augmentation', entity='cv-06')
        wandb.config.update(args)
        wandb.epochs = args.epochs
        wandb.run.name = save_name
        print(args)
        train(args, aug)
        wandb.finish()

    
    # wandb.run.name = folder_name
    
    # print(args)
    # train(args)
