#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 16:01
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : train_p_net.py
# @software: PyCharm

import argparse
import sys
import os
import config
from core.imagedb import ImageDB



def parse_args():
    parser = argparse.ArgumentParser(description='Train PNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(config.ANNO_STORE_DIR, config.PNET_TRAIN_IMGLIST_FILENAME),
                        help='training data annotation file', type=str)

    parser.add_argument('--model_path', dest='model_store_path', help='训练模型存储路径',
                        default=config.MODEL_STORE_DIR, type=str)

    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)

    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)

    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)

    parser.add_argument('--batch_size', dest='batch_size', help='训练Pnet批次大小',
                        default=config.TRAIN_BATCH_SIZE, type=int)

    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)


    parser.add_argument('--prefix_path', dest='',
                        help='training data annotation images prefix root path',
                        type=str)

    args = parser.parse_args()
    return args

def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=False):

    imagedb = ImageDB(annotation_file)

    gt_imdb = imagedb.load_imdb()
    # 这里是翻转进行数据增强，可以先不使用
    # gt_imdb = imagedb.append_flipped_images(gt_imdb)


    # train_pnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb,
    #            batch_size=batch_size, frequent=frequent, base_lr=lr, use_cuda=use_cuda)


if __name__ =="__main__":
    args = parse_args()
    print(args)

    train_net(args.annotation_file, args.model_store_path,
              end_epoch=args.end_epoch, frequent=args.frequent,
              lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
