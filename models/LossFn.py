#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 10:59
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : LossFn.py
# @software: PyCharm
import torch.nn as nn
import torch


class LossFn:
    def __init__(self, cls_facotr=1, box_factor=1, lamdmark_factor=1):
        self.cls_factor = cls_facotr
        self.box_factor = box_factor
        self.land_factor = lamdmark_factor
        self.loss_cls = nn.BCELoss()  # 二分类交叉熵
        self.loss_box = nn.MSELoss()  # 均方差
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, pred, label):
        pred_label = torch.squeeze(pred)
        gt_label = torch.squeeze(label)
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor

    def box_loss(self, pred_offset, gt_label, gt_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        # get the mask element with !=0 找到不等于0 的样本，不是负样本的索引
        unmask = torch.eq(gt_label, 0)
        mask = torch.eq(unmask, 0)

        # convert mask to dim index
        chose_index = torch.nonzero(mask.data)  # 输出非零元素的索引
        chose_index = torch.squeeze(chose_index)

        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor

    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label, -2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark) * self.land_factor
