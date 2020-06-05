#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
import torch
import argparse
import os, sys
import cv2
import numpy as np

if __name__ == '__main__':

    filename = '/home/yuxiang/GitLab/posecnn-pytorch/data/YCB_Object/models/industrial_dolly/industrialDolly_D_backup.png'
    im = cv2.imread(filename)

    filename = '/home/yuxiang/GitLab/posecnn-pytorch/data/YCB_Object/models/industrial_dolly/industrialDolly_D.png'
    im_new = im.copy()
    index = np.where(im[:,:,0] > 100)
    im_new[index[0], index[1], 0] -= 65
    im_new[index[0], index[1], 1] /= 2
    im_new[index[0], index[1], 2] /= 5
    im_new = np.clip(im_new, 0, 255)
    cv2.imwrite(filename, im_new)

    filename = '/home/yuxiang/GitLab/posecnn-pytorch/data/Images/isaac/6bd950e2-3d81-11ea-95d7-2d8ef6aa3e76.color.00000000000104928952966.png'
    target = cv2.imread(filename)

    # show images
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:,:,(2,1,0)])
    ax.set_title('original')

    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im_new[:,:,(2,1,0)])
    ax.set_title('changed')

    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(target[:,:,(2,1,0)])
    ax.set_title('target')

    plt.show()
