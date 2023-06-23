# python native
import os
import json
import random
import datetime
from functools import partial
import torch.nn.functional as F
import torchvision.transforms.functional as fn

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


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def dice_coef(y_true, y_pred):
    eps = 0.0001
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    ###
    torp = torch.sum(y_true_f,-1) + torch.sum(y_pred_f,-1) - intersection

    tn = (torp - torch.sum(y_pred_f,-1)) / torch.sum(y_true_f,-1)
    fp = (torp - torch.sum(y_true_f,-1)) / torch.sum(y_true_f,-1)
    ###
    
    return [(2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps),tn,fp]


def save_model(model, SAVED_DIR, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

    
def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    tns = []
    fps = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            #outputs = model(images)['out']
            outputs = model(images) #[0]
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            #outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice[0])
            tns.append(dice[1])
            fps.append(dice[2])
                
    dices = torch.cat(dices, 0)
    tns = torch.cat(tns, 0)
    fps = torch.cat(fps, 0)
    dices_per_class = torch.mean(dices, 0)
    tns_per_class = torch.mean(tns, 0)
    fps_per_class = torch.mean(fps, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f} TN {e.item():.4f} FP {f.item():.4f}"
        for c, d, e, f in zip(CLASSES, dices_per_class, tns_per_class, fps_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    avg_tns = torch.mean(tns_per_class).item()
    avg_fps = torch.mean(fps_per_class).item()
    
    return avg_dice ,avg_tns, avg_fps

def validation_mo(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    tns = []
    fps = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # images = fn.resize(images,1024)
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            #outputs = model(images)['out']
            outputs = model(images)[0]
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            #outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(masks, outputs)
            dices.append(dice[0])
            tns.append(dice[1])
            fps.append(dice[2])
                
    dices = torch.cat(dices, 0)
    tns = torch.cat(tns, 0)
    fps = torch.cat(fps, 0)
    dices_per_class = torch.mean(dices, 0)
    tns_per_class = torch.mean(tns, 0)
    fps_per_class = torch.mean(fps, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f} TN: {e.item():.4f} FP: {f.item():.4f}"
        for c, d, e, f in zip(CLASSES, dices_per_class, tns_per_class, fps_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    avg_tns = torch.mean(tns_per_class).item()
    avg_fps = torch.mean(fps_per_class).item()
    
    return avg_dice ,avg_tns, avg_fps


def increment_path(path, exist_ok=False):
    return f"{path}"

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred*target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2*intersection + smooth)/(pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()