#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 15:54
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : utils.py
# @software: PyCharm
import numpy as np

def IoU(box, boxes):
    """
    计算检测box和真实Boxes之间的IOU
    参数：box:narray,shape(5,),x1,y1,x2,y2,score
    boxes:narray,shape(n,4)，x1,y1,x2,y2，输入的gt boxes

    return : narray,shape(n,)，IOUS
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])  # 比较两个数组，并一个包含元素最小值的新数组。
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 计算bouding box width,height
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

if __name__ =="__main__":
    print("hello")
