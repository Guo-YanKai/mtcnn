#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 11:10
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : gen_Pnet_train_data.py
# @software: PyCharm

import sys
import numpy as np
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocess.utils import IoU


if __name__ =="__main__":
    anno_file = r"D:\code\work_code\github_code\mtcnn\train_data\trainImageList.txt"
    img_dir = r"D:\code\work_code\github_code\mtcnn\train_data\face_image"
    pos_save_dir = r"D:\code\work_code\github_code\mtcnn\train_data\positive"
    part_save_dir = r"D:\code\work_code\github_code\mtcnn\train_data\part"
    neg_save_dir = r'D:\code\work_code\github_code\mtcnn\train_data\negative'

    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num = len(annotations)
    print("%d pics in total" % num)

    p_idx = 0
    n_idx = 0
    d_idx = 0
    idx = 0
    box_idx = 0

    # store labels of positive, negative, part images
    f1 = open(os.path.join(r'D:\code\work_code\github_code\mtcnn\train_data\anno_store', 'pos_12.txt'), 'a')
    f2 = open(os.path.join(r'D:\code\work_code\github_code\mtcnn\train_data\anno_store', 'neg_12.txt'), 'a')
    f3 = open(os.path.join(r'D:\code\work_code\github_code\mtcnn\train_data\anno_store', 'part_12.txt'), 'a')
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        img_path = os.path.join(img_dir, annotation[0])
        img = cv2.imread(img_path)

        bbox = list(map(float, annotation[1:]))
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        boxes = np.array(bbox[:4], dtype=np.int32).reshape(-1, 4)

        idx += 1
        height, width, channels = img.shape
        print(height, width, channels)

        neg_num = 0
        while neg_num < 50:
            size = np.random.randint(12, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)

            crop_box = np.array([nx, ny, nx + size, ny + size])
            Iou = IoU(crop_box, boxes)

            cropped_img = img[ny: ny + size, nx: nx + size, :]
            resized_img = cv2.resize(cropped_img, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_img)
                n_idx += 1
                neg_num += 1

        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            # ignore small face,防止不准确
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # 生成negative example 同时根gt右overlop
            for i in range(5):
                size = np.random.randint(12, min(width, height) / 2)
                # x1,y1的offset为delta_x,delta_y
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)

                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IoU(crop_box, boxes)
                #             print("crop_box:", crop_box)
                #             print("Iou:",Iou)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + " 0\n")
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive example and part face
            for i in range(20):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))  # np.ceil计算大于等于该值的最小数
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                box_ = box.reshape(1, -1)

                if IoU(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

            box_idx += 1
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()