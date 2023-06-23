'''
input : 512
output : 2024
'''
# python native
import os

# external library
import albumentations as A
from torch.utils.data import Dataset, DataLoader


# torch
from torchinfo import summary
import torch.nn as nn

import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as fn
from dataset_all_2048 import *
from utils import set_seed, dice_coef, save_model, increment_path, dice_loss
from models import UNet_3Plus_2048 as UNet_3Plus
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss,IOU
from loss.msssimLoss import SSIM
import datetime
import pytz
import torch, gc
import argparse
import wandb
from tqdm.auto import tqdm
import torch.nn.functional as F

gc.collect()
torch.cuda.empty_cache()

### 저장 폴더명 생성
kst = pytz.timezone('Asia/Seoul')
now = datetime.datetime.now(kst)
folder_name = now.strftime('%Y-%m-%d-%H-%M-%S')

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
            images = fn.resize(images, 512)
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
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

def train(args):
    print(f'Start training..')

    set_seed(args.seed)

    SAVED_DIR = increment_path(os.path.join(args.model_dir, folder_name))
    SAVED_DIR += '_' + args.model
    print(SAVED_DIR)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    train_tf = A.Compose([A.HorizontalFlip(), A.GridDropout(ratio=0.1, random_offset=True, holes_number_x=4, holes_number_y=4, mask_fill_value=0), A.Rotate(limit=10, p=0.5)])
    valid_tf = None
    train_dataset = XRayDataset_all_2048(is_train=True, transforms=train_tf)
    valid_dataset = XRayDataset_all_2048(is_train=False, transforms=valid_tf)
    
    wandb.init(project='project3_Group_KFold', entity='cv-06')
    wandb.config.update(args)
    wandb.epochs = args.epochs
    wandb.run.name = folder_name + '_' + 'fold2'

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    model = UNet_3Plus.UNet_3Plus_DeepSup(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)
    summary(model.cuda(), input_size=(1,3, 512, 512))
    if args.pre_weight !='':
        pre_model = torch.load(args.pre_weight)
        
        model_st = pre_model.state_dict()
        model.load_state_dict(model_st, strict=False)
        
    # Loss function 정의
    criterion = nn.BCELoss()#nn.BCEWithLogitsLoss()
    criterion_b = nn.BCELoss()#reduction='mean')
    criterion_i = dice_loss#IOU()
    criterion_m = SSIM()
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
            images = fn.resize(images, 512)
            
            # inference
            outputs = model(images)
            if step==0:
                print(outputs[0].shape)
                print(outputs[4].shape)

            # loss 계산
            loss1_i = criterion_i(outputs[0], masks)
            loss1_b = criterion_b(outputs[0], masks)
            loss2_i = criterion_i(outputs[1], masks)
            loss2_b = criterion_b(outputs[1], masks)
            loss3_i = criterion_i(outputs[2], masks)
            loss3_b = criterion_b(outputs[2], masks)
            loss4_i = criterion_i(outputs[3], masks)
            loss4_b = criterion_b(outputs[3], masks)
            loss5_i = criterion_i(outputs[4], masks)
            loss5_b = criterion_b(outputs[4], masks)

            ##--------------------------
            optimizer.zero_grad()

            loss = (loss1_i + loss1_b +loss2_i +loss2_b +loss3_i +loss3_b +loss4_i +loss4_b)/5

            loss.backward()
            optimizer.step()
            wandb.log({'train_bce_iou_loss': loss})

        if not os.path.isdir(SAVED_DIR):
            os.makedirs(SAVED_DIR)

        if (epoch) % 1 == 0:
            save_model(model, SAVED_DIR, file_name=f'{epoch}.pt')
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch) % args.interval == 0:
            dice = validation_mo(epoch, model, valid_loader, criterion, args.thr)
            wandb.log({'val_dice': dice[0]})
            wandb.log({'val_TN': dice[1]})
            wandb.log({'val_FP': dice[2]})
            
            if best_dice < dice[0]:
                print(f"Best performance at epoch: {epoch}, {best_dice:.4f} -> {dice[0]:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice[0]
                save_model(model, SAVED_DIR,file_name='best.pt')
        
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')########
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 1)')########
    parser.add_argument('--dataset', type=str, default='XRayDataset_all_2048', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='half_filter', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='unet3_DeepSup_2048', help='model type')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')#####
    parser.add_argument('--criterion', type=str, default='BCEWithLogitsLoss', help='criterion type (default: focal)')
    parser.add_argument('--interval', type=int, default=1, help='batch intervals for validation')########
    parser.add_argument('--thr', type=float, default=0.5, help='valid thr')########
    parser.add_argument('--pre_weight', type=str, default='./trained_model/2023-06-21-04-24-18_unet3_DeepSup_2048/30.pt', help='for transfer learning')########
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/DCM'))########
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './trained_model'))########
    args = parser.parse_args()

    print(args)
    train(args)
