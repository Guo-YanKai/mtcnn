#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 17:33
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : image_reader.py
# @software: PyCharm

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler

def get_minibatch(imdb):

    # img_size:12,24,or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    landmark_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]["image"])
        if imdb[i]["flipped"]:
            im = im[:,::-1,:]

        cls = imdb[i]["label"]
        bbox_target = imdb[i]["bbox_target"]
        landmark = imdb[i]["landmark_target"]

        processed_ims.append(im)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)
        landmark_reg_target.append(landmark)
    im_array = np.asarray(processed_ims)
    label_array = np.array(cls_label) # 按照行放在一起
    bbox_reg_target = np.vstack(bbox_reg_target)
    landmark_reg_target = np.vstack(landmark_reg_target)
    data = {"data":im_array}
    label = {"label":label_array,
             "bbox_target":bbox_reg_target,
             "landmark_target":landmark_reg_target}
    return data, label



class TrainImageReader:
    def __init__(self, imdb, im_size, batch_size=128, shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target', 'landmark_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data,self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = get_minibatch(imdb)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]


def split_data_val(dataset, valid_rate, shuffle=True):
    """打乱数据，划分验证集
        参数：dataset：实例化后的Dataset对象
            args: 超参数
            shuffle:是否shuffle数据"""
    print("total sample:", len(dataset))

    data_size = len(dataset)
    indices = list(range(data_size))  # 生成索引
    split = int(np.floor(valid_rate * data_size))  # np.floor返回不大于输入参数的最大整数
    if shuffle:
        np.random.seed(2022)
        np.random.shuffle(indices)  # 根据随机种子打散索引
    train_indices, val_indices = indices[split:], indices[:split]

    # 生成数据采样器和加载器
    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    print(f"train sample: {len(train_indices)},  val sample: {len(val_indices)}")
    return train_sample, val_sample


class LanMarkDataset(Dataset):
    def __init__(self, imdb):
        super(LanMarkDataset, self).__init__()
        self.imdb = imdb

        self.transforme = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.imdb)


    def __getitem__(self, item):
        img = cv2.imread(self.imdb[item]["image"])
        img = self.transforme(img)

        label = self.imdb[item]["label"]
        bbox_target = self.imdb[item]["bbox_target"]
        landmark_target = self.imdb[item]["landmark_target"]

        # print("label:",label,type(label))
        # print("bbox_target:",bbox_target, type(bbox_target))
        # print("landmark_target:",landmark_target,type(landmark_target))

        data = {"image": img}
        label = {"label": label,
                 "bbox_target": bbox_target,
                 "landmark_target": landmark_target}

        return data, label






