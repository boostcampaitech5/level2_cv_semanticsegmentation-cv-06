# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2
from .init_weights import init_weights

'''
    High_Resol
    input = 29*1024*1024  (0~1)
    output = 29*2048*2048 (0~1) 
'''
class High_Resol(nn.Module):
    def __init__(self, in_channels=29, inplace=True):
        super(High_Resol,self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)#cls*2 2048
        self.layer2 = nn.BatchNorm2d(2*in_channels)
        self.layer3 = nn.ReLU(inplace=inplace)

        self.layer4 = nn.Conv2d(in_channels=2*in_channels, out_channels=4*in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)#cls*4 2048
        self.layer5 = nn.BatchNorm2d(4*in_channels)
        self.layer6 = nn.ReLU(inplace=inplace)

        self.layer7 = nn.Conv2d(in_channels=4*in_channels, out_channels=2*in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)#cls*2 2048
        self.layer8 = nn.BatchNorm2d(2*in_channels)
        self.layer9 = nn.ReLU(inplace=inplace)

        self.layer10 = nn.Conv2d(in_channels=2*in_channels, out_channels=1*in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)#cls*1 2048




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self,input):
        feature = F.interpolate(input, size=(2048, 2048), mode="bilinear")
        
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        feature = self.layer5(feature)
        feature = self.layer6(feature)
        feature = self.layer7(feature)
        feature = self.layer8(feature)
        feature = self.layer9(feature)
        feature = self.layer10(feature)

        feature = F.sigmoid(feature)

        return feature

