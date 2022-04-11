#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 17:20
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : train_tools.py
# @software: PyCharm

import os
from models.models import PNet
import torch
from core.image_reader import LanMarkDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from core.logger import Train_Logger, LossAverage, AccAverage
from tqdm import tqdm
from models.LossFn import LossFn
from collections import OrderedDict


def val_pnet(net, val_loader, criterion, device):
    net.eval()
    val_loss = LossAverage()
    val_acc = AccAverage()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            image_tensor = batch[0]["image"].to(device, dtype=torch.float32)
            gt_label = batch[1]["label"].to(device, dtype=torch.float32)
            gt_bbox = batch[1]["bbox_target"].to(device, dtype=torch.float32)
            # 训练Pnet不需要关键点
            # gt_landmark = batch[1]["landmark_target"].to(device,dtype=torch.float32)
            cls_prod, box_offset_prod = net(image_tensor)
            cls_loss = criterion.cls_loss(cls_prod, gt_label)
            box_offset_loss = criterion.box_loss(box_offset_prod, gt_label, gt_bbox)
            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            val_loss.update(all_loss.item(), image_tensor.size(0))
            val_acc.update(cls_prod, gt_label)

        val_log = OrderedDict({"Val_Loss": val_loss.avg})
        val_log.update({"Val_acc": val_acc.avg})
    return val_log


def train_pnet(model_store_path, end_epoch, imdb,
               batch_size, base_lr=0.01, use_cuda=True):
    device = torch.device("cuda:1")
    os.makedirs(model_store_path, exist_ok=True)
    net = PNet(is_train=True, use_cuda=use_cuda).to(device)
    net.train()

    criterion = LossFn()
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

    dataset = LanMarkDataset(imdb)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=batch_size, num_workers=0,
                              pin_memory=True, shuffle=False)

    val_loader = DataLoader(val, batch_size=batch_size, num_workers=0,
                            pin_memory=True, shuffle=False)

    log = Train_Logger(model_store_path, "train_Pnet_log")
    best = [0, float("inf"), float("inf")]
    trigger = 0

    for cur_epoch in range(1, end_epoch + 1):
        train_loss = LossAverage()
        train_acc = AccAverage()
        print("=====Epoch:{}======lr:{}".format(cur_epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            image_tensor = batch[0]["image"].to(device, dtype=torch.float32)
            gt_label = batch[1]["label"].to(device, dtype=torch.float32)
            gt_bbox = batch[1]["bbox_target"].to(device, dtype=torch.float32)

            # 训练Pnet不需要关键点
            # gt_landmark = batch[1]["landmark_target"].to(device,dtype=torch.float32)

            # print("image_tensor:", image_tensor)
            # print("gt_label:", gt_label, gt_label.shape)
            # print("gt_bbox:", gt_bbox, gt_bbox.shape)
            # print("gt_landmark:", gt_landmark)

            cls_pred, box_offset_pred = net(image_tensor)

            cls_loss = criterion.cls_loss(cls_pred, gt_label)
            box_offset_loss = criterion.box_loss(box_offset_pred, gt_label, gt_bbox)
            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            all_loss.backward()
            optimizer.step()

            train_loss.update(all_loss.item(), cls_pred.shape[0])
            train_acc.update(cls_pred, gt_label)

        train_log = OrderedDict({"Train_Loss": train_loss.avg})
        train_log.update({"Train_acc": train_acc.avg})
        train_log.update({"lr": optimizer.state_dict()["param_groups"][0]["lr"]})

        # 验证过程
        val_log = val_pnet(net, val_loader, criterion, device)
        scheduler.step(val_log["Val_Loss"])

        log.update(cur_epoch, train_log, val_log)
        # save checkpoints
        state = {"net": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": cur_epoch}
        torch.save(state, os.path.join(model_store_path, "latest_model.pth"))
        trigger += 1

        if val_log["Val_Loss"] < best[1]:
            print("save best model")
            torch.save(state, os.path.join(model_store_path, "best_model.pth"))
            best[0] = cur_epoch
            best[1] = val_log["Val_Loss"]
            best[2] = val_log["Val_acc"]
            trigger = 0
        print("Best Performance at Epoch:{}|{}".format(best[0], best[1]))
        # 早停
        if trigger >= 20:
            print("=>early stopping")
            break
    torch.cuda.empty_cache()
