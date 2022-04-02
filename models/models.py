#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 17:26
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : models.py
# @software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

def weigths_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)


class PNet(nn.Module):
    def __init__(self, is_train=True, use_cuda=True):
        super(PNet, self).__init__()

        self.is_train = is_train
        self.use_cuda = use_cuda

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
            nn.PReLU(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
            nn.PReLU(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.PReLU(32)
        )
        # detection
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

        # bounding box regresion
        self.conv4_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)

        # landmark localization 原输出5个关键，out_channels=10,现输出6个关键，out_channels=12
        self.conv4_3 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1, stride=1)

        # weigth init with xavier
        self.apply(weigths_init)


    def forward(self, x):
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        # landmark = self.conv4_3(x)

        if self.is_train is True:
            return label, offset
        return label, offset


class RNet(nn.Module):
    def __init__(self, is_train=False, use_cuda =True):
        super(RNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1),
            nn.PReLU(28),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),
            nn.PReLU(48),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=48, out_channels=64,  kernel_size=2, stride=1),
            nn.PReLU(64)
        )

        self.conv4 = nn.Linear(in_features=64*3*3, out_features=128)



    def forward(self,x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        return x



if __name__ == "__main__":
    x = torch.randn((2, 3, 24, 24))
    net = RNet()
    # print("label:",net(x)[0].shape)
    # print("offset:", net(x)[1].shape)

    print(net(x).shape)
