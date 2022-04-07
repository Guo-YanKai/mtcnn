#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 16:31
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : imagedb.py
# @software: PyCharm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ImageDB(object):

    def __init__(self, image_annotation_file, prefix_path="", mode="train"):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file

        self.classes = ["__background__", "face"]
        self.num_classes = 2

        self.image_set_index = self.load_image_set_idx()
        self.num_images = len(self.image_set_index)
        self.mode = mode
        print("self.num_images:", self.num_images)

    def load_image_set_idx(self):
        assert os.path.exists(self.image_annotation_file), f"path dose not exist：{self.image_annotation_file}"
        with open(self.image_annotation_file, "r") as f:
            image_set_idx = [x.strip().split(" ") for x in f.readlines()]
        return image_set_idx

    def load_imdb(self):
        """获取并保存真实gt的图像数据库，返回字典类型"""
        gt_imdb = self.load_annotations()
        return gt_imdb

    def load_annotations(self, anntion_type=1):
        assert os.path.exists(self.image_annotation_file), \
            f"annotations not found at {self.image_annotation_file}"
        with open(self.image_annotation_file, "r") as f:
            annotations = f.readlines()
        imdb = []
        for i in range(self.num_images):
            annotation = annotations[i].strip().split(" ")
            index = annotation[0]
            img_path = self.real_image_path(index)
            imdb_ = dict()
            imdb_["image"] = img_path

            if self.mode == "test":
                pass
            else:
                label = annotation[1]
                imdb_["label"] = int(label)
                imdb_["flipped"] = False
                imdb_["bbox_target"] = np.zeros((4,))
                imdb_["landmark_target"] = np.zeros((12,))
                if len(annotation[2:])==4:
                    bbox_target = annotation[2:6]
                    imdb_["bbox_target"] = np.array(bbox_target).astype(float)
                if len(annotation[2:])==16:
                    bbox_target = annotation[2:6]
                    imdb_["bbox_target"] = np.array(bbox_target).astype(float)
                    landmark  = annotation[6:]
                    imdb_["landmark_target"] = np.array(landmark).astype(float)

            imdb.append(imdb_)
        return imdb

    def real_image_path(self, index):
        """给定图片的index,返回图片path"""
        index = index.replace("\\", "/")
        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file = index

        # 这里不需要判断
        # if not image_file.endswith(('jpg','png','jpeg','bmp')):
        #     image_file = image_file + '.jpg'

        assert os.path.exists(image_file), f"path does not exists {image_file}"

        return image_file


    def append_flipped_images(self,imdb):
        """
        阔以先不使用
        :param imdb:原始图像训练集
        :return: 将翻转后的图像添加到Imdb
        """
        print("append flipped images to imdb", len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_["bbox_target"].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            landmark_ =  imdb_["landmark_target"].copy()
            landmark_ = landmark_.reshape((6,2))
            landmark_ = np.asarray([(1-x, y) for (x,y) in landmark_])
            landmark_[[0,1]] = landmark_[[1,0]]
            landmark_[[3,4]] = landmark_[[4,3]]

            item = {'image': imdb_['image'],
                    'label': imdb_['label'],
                    'bbox_target': m_bbox,
                    'landmark_target': landmark_.reshape((10)),
                    'flipped': True}
            imdb.append(item)

        self.image_set_index *=2

        return imdb
