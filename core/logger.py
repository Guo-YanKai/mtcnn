#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 11:56
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : logger.py
# @software: PyCharm

from collections import OrderedDict
import numpy as np
import torch, random
from tensorboardX import SummaryWriter
import pandas as pd
import os


def dict_round(dic, num):
    """将dic中的值取num位小数"""
    for k, v in dic.items():
        if k == "lr":
            dic[k] = round(v, num * 2)
        else:
            dic[k] = round(v, num)
    return dic


class Train_Logger():
    """保存训练过程中的各种指标，csv保存、tensorboard可视化"""

    def __init__(self, save_path, save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update(self, epoch, train_log, val_log):
        # 有序字典
        item = OrderedDict({"epoch": epoch})
        item.update(train_log)
        item.update(val_log)
        item = dict_round(item, 4)
        print("\033[0:32m Train: \033[0m", train_log)
        print("\033[0:32m Val: \033[0m", val_log)
        self.updata_csv(item)
        self.updata_tensorboard(item)

    def updata_csv(self, item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log.append(item, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv("%s/%s.csv" % (self.save_path, self.save_name), index=False)

    def updata_tensorboard(self, item):
        if self.summary is None:
            self.summary = SummaryWriter("%s/" % (self.save_path))
        epoch = item["epoch"]
        for key, value in item.items():
            if key != "epoch":
                self.summary.add_scalar(key, value, epoch)


class LossAverage(object):
    """计算并存储当前损失值和平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # we only need the detection which >= 0
    mask = torch.ge(gt_cls, 0)
    # get valid element
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    ## if size == 0 meaning that your gt_labels are all negative, landmark or part
    ## divided by zero meaning that your gt_labels are all negative, landmark or part
    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


class AccAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cls_prod, cls_gt):
        self.val = compute_accuracy(cls_prod, cls_gt).data.cpu().numpy()
        self.sum += self.val * len(cls_gt)
        self.count += len(cls_gt)
        self.avg = round(self.sum / self.count, 4)
