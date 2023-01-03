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
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
def output_block():
    Layer = nn.Sequential(
        nn.Conv2d(in_channels=32,out_channels=1,kernel_size=(1,1)),
        nn.Sigmoid()
    )
    return Layer
class DoubleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = VGGBlock(3,64,64)
        self.conv2 = VGGBlock(64,128,128)
        self.conv3 = VGGBlock(128,256,256)
        self.conv4 = VGGBlock(256,512,512)
        self.conv5 = VGGBlock(512,512,512)

        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.Vgg1 = VGGBlock(1024,256,256)
        self.Vgg2 = VGGBlock(512,128,128)
        self.Vgg3 = VGGBlock(256,64,64)
        self.Vgg4 = VGGBlock(128,32,32)

        self.out = output_block()

        self.conv11 = VGGBlock(6,32,32)
        self.conv12 = VGGBlock(32,64,64)
        self.conv13 = VGGBlock(64,128,128)
        self.conv14 = VGGBlock(128,256,256)

        self.Vgg5 = VGGBlock(1024,256,256)
        self.Vgg6 = VGGBlock(640,128,128)
        self.Vgg7 = VGGBlock(320,64,64)
        self.Vgg8 = VGGBlock(160,32,32)

        self.out1 = nn.Conv2d(in_channels=32,out_channels=1,kernel_size=(1,1))
    def forward(self,x):
        x1 = self.conv1(x)

        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        x5 = self.up(x5)
        x5  = torch.cat([x5,x4],1)
        x6 = self.Vgg1(x5)

        x6 = self.up(x6)
        x6 = torch.cat([x6,x3],1)
        x7 = self.Vgg2(x6)

        x7 = self.up(x7)
        x7 = torch.cat([x7,x2],1)
        x8 = self.Vgg3(x7)

        x8 = self.up(x8)
        x8 = torch.cat([x8,x1],1)
        x9 = self.Vgg4(x8)

        output1 = self.out(x9)
        output1 = x*output1
        x = torch.cat([x,output1],1)
        x11 = self.conv11(x)
        x12 = self.conv12(self.pool(x11))
        x13 = self.conv13(self.pool(x12))
        x14 = self.conv14(self.pool(x13))

        y = self.pool(x14)

        y = self.up(y)
        y = torch.cat([y,x14,x4],1)
        y = self.Vgg5(y)


        y = self.up(y)
        y = torch.cat([y,x13,x3],1)
        y = self.Vgg6(y)

        y = self.up(y)
        y = torch.cat([y,x12,x2],1)
        y = self.Vgg7(y)

        y = self.up(y)
        y = torch.cat([y,x11,x1],1)
        y = self.Vgg8(y)

        output2 = self.out1(y)
        return output2


model = DoubleUNet()
torch.save(model.state_dict(),'doubleunet.pth')
x = torch.randn(1,3,128,128)

print(model(x).size())