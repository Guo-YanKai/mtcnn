#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 16:45
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : assemble_pnet_imglist.py
# @software: PyCharm

import os
import os
import numpy.random as npr
import numpy as np


def assemble_data(output_file, anno_file_list=[]):
    size = 12
    if len(anno_file_list) == 0:
        return 0
    if os.path.exists(output_file):
        os.remove(output_file)
    chose_count = 0
    for anno_file in anno_file_list:
        with open(anno_file, "r") as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000
        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > 100000:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))

        with open(output_file, "a+") as f:
            for idx in idx_keep:
                f.write(anno_lines[idx])
                chose_count += 1
    return chose_count

if __name__ =="__main__":
    pnet_postive_file = '/data/dl/gyk/mtcnn/train_data/anno_store/pos_12.txt'
    pnet_part_file = '/data/dl/gyk/mtcnn/train_data/anno_store/part_12.txt'
    pnet_neg_file = '/data/dl/gyk/mtcnn/train_data/anno_store/neg_12.txt'
    imglist_filename = '/data/dl/gyk/mtcnn/train_data/anno_store/imglist_anno_12.txt'
    anno_list = []
    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    chose_count = assemble_data(imglist_filename, anno_list)
    print(chose_count)