#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 12:38
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : config.py
# @software: PyCharm

import os


MODEL_STORE_DIR = os.path.dirname(os.path.realpath(__file__))+"/model_store"

ANNO_STORE_DIR = os.path.dirname(os.path.realpath(__file__))+"/train_data/anno_store"

LOG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/log"

USE_CUDA = True

TRAIN_BATCH_SIZE = 2

TRAIN_LR = 0.0001

END_EPOCH = 200

PNET_POSTIVE_ANNO_FILENAME = "pos_12.txt"
PNET_NEGATIVE_ANNO_FILENAME = "neg_12.txt"
PNET_PART_ANNO_FILENAME = "part_12.txt"
PNET_LANDMARK_ANNO_FILENAME = "landmark_12.txt"


RNET_POSTIVE_ANNO_FILENAME = "pos_24.txt"
RNET_NEGATIVE_ANNO_FILENAME = "neg_24.txt"
RNET_PART_ANNO_FILENAME = "part_24.txt"
RNET_LANDMARK_ANNO_FILENAME = "landmark_24.txt"


ONET_POSTIVE_ANNO_FILENAME = "pos_48.txt"
ONET_NEGATIVE_ANNO_FILENAME = "neg_48.txt"
ONET_PART_ANNO_FILENAME = "part_48.txt"
ONET_LANDMARK_ANNO_FILENAME = "landmark_48.txt"

PNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_12.txt"
RNET_TRAIN_IMGLIST_FILENAME = "imglist_anno_24.txt"
ONET_TRAIN_IMGLIST_FILENAME = "imglist_anno_48.txt"

if __name__ =="__main__":
    print(ANNO_STORE_DIR)

