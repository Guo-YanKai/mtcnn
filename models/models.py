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
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


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
    def __init__(self, is_train=False, use_cuda=True):
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
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),
            nn.PReLU(64)
        )

        self.conv4 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        self.prelu4 = nn.PReLU(128)

        # detection
        self.conv5_1 = nn.Linear(128, 1)

        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)

        # landmark localization
        self.conv5_3 = nn.Linear(128, 12)

        # weight initiation weight xavier
        self.apply(weigths_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)

        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)

        # landmark
        # landmark = self.conv5_3(x)

        if self.is_train is True:
            return det, box
        # landamark = self.conv5_3(x)

        return det, box


class ONet(nn.Module):
    def __init__(self, is_train=False, use_cuda=True):
        super(ONet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda

        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(64),
            nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.PReLU(128),
        )
        self.conv5 = nn.Linear(128*3*3,out_features=256)
        self.prelu5 = nn.PReLU(256)

        # detection
        self.conv6_1= nn.Linear(in_features=256, out_features=1)
        # bounding box
        self.conv6_2 =nn.Linear(in_features=256, out_features=4)
        # landmark locatiation
        self.conv6_3 = nn.Linear(in_features=256, out_features=12)

        # initial weight
        self.apply(weigths_init)


    def forward(self, x):
        x = self.pre_layer(x)

        x = x.view(x.size(0),-1)
        x = self.conv5(x)
        x = self.prelu5(x)

        det = torch.sigmoid(self.conv6_1(x))

        box = self.conv6_2(x)
        landmark = self.conv6_3(x)

        if self.is_train is True:
            return det, box, landmark
        return det, box, landmark



if __name__ == "__main__":
    x = torch.randn((2, 3, 12,12))
    net = PNet()
    # print(net(x).shape)
    print(net(x)[0].shape)
    print(net(x)[1].shape)
    # print(net(x)[2].shape)