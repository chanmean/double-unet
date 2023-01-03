import os
from glob import glob
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split

img_size = 128

config ={}
config['name'] = None
config['epochs']=100
config['batch_size'] =16

config['arch'] = 'NestdUNet'
config['deep_supervision'] = False
config['input_channels'] =3
config['num_classes']=1
config['input_w']=128
config['input_h']=128

config['loss'] ='BCEDiceLoss'

config['dataset']=''
config['img_ext'] ='.png'
config['mask_ext']='.png'

config['optimizer']='SGD'
config['lr']=1e-3
config['weight_decay'] = 1e-4
config['momentum']=0.9
config['nesterov']=False

config['scheduler'] = 'CosineAnnealingLR'
config['min_lr']=1e-5
config['factor']=0.1
config['patience']=2
config['milestones']='1,2'
config['early_stopping']=-1
config['num_workers']=4

class Squeeze_Excite(nn.Module):
    def __init__(self,channels,reduction):
        super().__init__()
        self.avgpool = nn.AdaptivePool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels,bias=False),
            nn.Sigmoid()

        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y= self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)
class VGGBlock(nn.Module):
    def __init__(self,in_channels,middle_channels,out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,middle_channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels,out_channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels,8)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)
        return(out)