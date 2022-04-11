#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 11:11
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : detect.py.py
# @software: PyCharm

import cv2
import time
import numpy as np
import torch
from torch.autograd.variable import Variable
from models.models import PNet, RNet, ONet
from core.image_reader import convert_image_to_tensor
from core.image_tools import  convert_chwTensor_to_hwcNumpy

def creat_mtcnn_net(p_model_path=None, r_model_path=None,
                    o_model_path=None, use_cuda=True):
    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if (use_cuda):

            device = torch.device("cuda:0")
            ckpt = torch.load("{}/best_model.pth".format(p_model_path), map_location=device)
            pnet.load_state_dict(ckpt["net"])
        else:
            device = torch.device("cpu")
            ckpt = torch.load("{}/best_model.pth".format(p_model_path), map_location=device)
            pnet.load_state_dict(ckpt["net"])
        print("pnet Model loaded！")

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            device = torch.device("cuda:0")
            ckpt = torch.load("{}/best_model.pth".format(r_model_path), map_location=device)
            rnet.load_state_dict(ckpt["net"])
        else:
            device = torch.device("cpu")
            ckpt = torch.load("{}\\best_model.pth".format(r_model_path), map_location=device)
            rnet.load_state_dict(ckpt["net"])
        print("rnet Model loaded！")

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            device = torch.device("cuda:0")
            ckpt = torch.load("{}\\best_model.pth".format(o_model_path), map_location=device)
            onet.load_state_dict(ckpt["net"])
        else:
            device = torch.device("cpu")
            ckpt = torch.load("{}\\best_model.pth".format(o_model_path), map_location=device)
            onet.load_state_dict(ckpt["net"])
        print("onet Model loaded！")

    return pnet, onet, rnet


class MtcnnDetector(object):
    """P,R,O net face detection（人脸检测） and landmark align(关键点排列)"""

    def __init__(self, pnet=None, rnet=None, onet=None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet

        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
    def unique_image_fromat(self, im):
        # 统一图片格式
        if not isinstance(im, np.ndarray):
            if im.mode == "I":
                im = np.array(im, np.int32, copy=False)
            elif im.mode == "I;16":
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    def square_bbox(self, bbox):
        """
        转换为方形的bbox
        :param bbox: np.array,shape:n*m 个inputbbox
        :return: 方形bbox
        """

        square_bbox = bbox.copy()  # 返回数组的副本
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1

        l = np.maximum(h, w)

        # x1 = x1 + w*0.5- l*0.5
        # y1 = y1 + w*0.5 -l*0.5

        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1

        return square_bbox

    def generate_bounding_box(self, map, reg, scale, threshold):
        """
        未完成！！！！！！！！
        从feature map中生成bbox
        :param map: numpy.array,shape:[n,m,1]
        :param reg: numpy.array, shape:[n,m,4]
        :param scale: float number, 检测的scale
        :param threshold: float number, 检测阈值
        :return: bbox array
        """
        stride = 2
        callsize = 12  # 感受野
        t_index = np.where(map > threshold)

    def resize_image(self, img, scale):
        """
            resize image and transform dimention to [batchsize, channel, height, width]
        Parameters:
        ----------
            img: numpy array , height x width x channel
                input image, channels in BGR order here
            scale: float number
                scale factor of resize operation
        Returns:
        -------
            transformed image tensor , 1 x channel x height x width
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def detect_pnet(self, im):
        """
        get face condidates through pnet
        :param im: numpy.array, input image array, one batch
        :return:
            bboxes:numpy.array,校准(calibration)前检测的bboxes
            bboxes_algin:numpy array校准后的bboxes
        """

        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size
        print('img shape:{0}, current_scale:{1}'.format(im.shape, current_scale))

        im_resized = self.resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape


        all_boxes= list()
        i = 0

        while min(current_height, current_width) > net_size:
            feed_imgs = []
            print("im_resized:", im_resized.shape, type(im_resized))
            image_tensor = convert_image_to_tensor(im_resized)
            print("image_tensor:", image_tensor.shape, type(image_tensor))

            feed_imgs.append(image_tensor)


            feed_imgs = torch.vstack(feed_imgs)
            feed_imgs = Variable(feed_imgs)
            feed_imgs = torch.unsqueeze(feed_imgs, 0)
            print("feed_imgs:", feed_imgs.shape, type(feed_imgs))

            # device = torch.device("cuda:0")
            # feed_imgs = feed_imgs.to(device, dtype=torch.float32)

            cls_map, reg = self.pnet_detector(feed_imgs)
            print("cls_map:", cls_map.shape, "reg:", reg.shape)

            cls_map_np = convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            print("cls_map_np:", cls_map_np.shape)

            reg_np = convert_chwTensor_to_hwcNumpy(reg.cpu())
            print("reg_np:", reg_np.shape)

            # ---------------------gyk待看----------------------
            boxes = self.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

            break

        return 1, 2