'''
input : 512
output : 512
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
from torchvision import transforms
import torchvision.transforms.functional as fn
from dataset import *
from utils import set_seed, dice_coef, save_model, validation, validation_mo, increment_path, dice_loss
from models import UNet_3Plus
from loss.bceLoss import BCE_loss
from loss.iouLoss import IOU_loss,IOU
from loss.msssimLoss import SSIM
import datetime
import pytz
import torch, gc
import argparse
import wandb
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

def train(args):
    print(f'Start training..')

    set_seed(args.seed)

    SAVED_DIR = increment_path(os.path.join(args.model_dir, save_name))
    print(SAVED_DIR)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    train_tf = A.Compose([A.Resize(512, 512), A.HorizontalFlip(), A.GridDropout(ratio=0.1, random_offset=True, holes_number_x=4, holes_number_y=4), A.Rotate(limit=45, p=0.5)])
    valid_tf = A.Compose([A.Resize(512, 512)])
    
    train_dataset = XRayDataset_all(is_train=True, transforms=train_tf)
    valid_dataset = XRayDataset_all(is_train=False, transforms=valid_tf)

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
    
    model = UNet_3Plus.UNet_3Plus_DeepSup(in_channels=3, n_classes=29, feature_scale=4, is_deconv=True, is_batchnorm=True)
    summary(model.cuda(), input_size=(2, 3, 512, 512))
    if args.pre_weight !='':
        model = torch.load(args.pre_weight)
    
    # Loss function 정의
    criterion = nn.BCELoss()#nn.BCEWithLogitsLoss()
    criterion_b = nn.BCELoss()#reduction='mean')
    criterion_i = dice_loss#IOU()

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

        if epoch % args.interval == 0:
            save_model(model, SAVED_DIR, file_name=f'{epoch}.pt')
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch) % args.interval == 0:
            dice = validation_mo(epoch, model, valid_loader, criterion, args.thr)
            print(f"epoch {epoch}: {dice[0]:.4f}")
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
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='XRayDataset_all', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='half_filter', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='unet3_DeepSup', help='model type (default: unet3)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='BCEWithLogitsLoss', help='criterion type (default: focal)')
    parser.add_argument('--interval', type=int, default=1, help='batch intervals for validation')
    parser.add_argument('--thr', type=float, default=0.45, help='valid thr')
    parser.add_argument('--pre_weight', type=str, default='', help='for transfer learning')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/DCM'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './trained_model'))
    args = parser.parse_args()
    
    save_name = folder_name +  '-' + 'fold2'
    wandb.init(project='project3_Betty_augmentation', entity='cv-06')
    wandb.config.update(args)
    wandb.epochs = args.epochs
    wandb.run.name = save_name + '_Rotate45'
    print(args)
    train(args)
    wandb.finish()