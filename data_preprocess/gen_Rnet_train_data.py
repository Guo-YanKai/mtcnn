#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 10:51
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : gen_Rnet_train_data.py
# @software: PyCharm


import argparse

import cv2
import numpy as np
import sys
import os
import torch
from core.detect import creat_mtcnn_net,MtcnnDetector
from core.imagedb import ImageDB
from torch.utils.data import DataLoader
from core.image_reader import TestDataset, TestImageLoader
from tqdm import tqdm
import time

def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path,
                  use_cuda=True, vis=False):
    """
    :param data_dir: 训练数据
    :param anno_file:
    :param pnet_model_file:
    :param prefix_path:
    :param use_cuda:
    :param vis:
    :return:
    """
    print("data_dir:", data_dir)
    print("anno_file: ", anno_file)
    print("pnet_model_file:", pnet_model_file)

    # 导入训练好的Pnet
    pnet, _, _ = creat_mtcnn_net(p_model_path=pnet_model_file, use_cuda=True)



    mtcnn_detector = MtcnnDetector(pnet=pnet, min_face_size=12) # 尚未完成

    imagedb = ImageDB(image_annotation_file=anno_file,prefix_path=prefix_path, mode="test")
    imdb = imagedb.load_imdb()


    # testdataset = TestDataset(imdb)
    # test_laoder = DataLoader(testdataset, batch_size=1, num_workers=0,
    #                          pin_memory=True, shuffle=False)

    image_reader = TestImageLoader(imdb=imdb, batch_size=1, shuffle=False)

    all_boxes = list()
    batch_idx = 0

    for databatch  in image_reader:

        if batch_idx%100==0:
            print("%d images done"%batch_idx)
        t = time.time()
        print("time:", t)

        img = databatch

        boxes, boxes_align  = mtcnn_detector.detect_pnet(im=img)



        break








if __name__ == "__main__":
    # 保存Rnet训练数据的路径
    traindata_store = r"D:\code\work_code\github_code\mtcnn\train_data\anno_store"

    # 原始训练数据
    annotaion_file = r"D:\code\work_code\github_code\mtcnn\train_data\trainImageList.txt"

    # Pnet模型保存路径
    pnet_model_file = r"D:\code\work_code\github_code\mtcnn\model_store\Pnet"
    prefix_path = r"D:\code\work_code\github_code\mtcnn\train_data\face_image"
    use_cuda = True

    gen_rnet_data(traindata_store, annotaion_file, pnet_model_file, prefix_path, use_cuda)
    torch.cuda.empty_cache()
