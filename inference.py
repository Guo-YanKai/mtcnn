#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 11:01
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : inference.py
# @software: PyCharm

import cv2





if __name__ =="__main__":
    image_path = '1000454P1.png'
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(img)
    print(detections)