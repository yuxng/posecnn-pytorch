#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
import argparse
import os, sys
from transforms3d.quaternions import mat2quat, quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
from utils.se3 import *
import libsynthesizer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    height = 480
    width = 640
    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    zfar = 6.0
    znear = 0.25
    num_classes = 22
    factor_depth = 10000.0
    root = '/home/yuxiang/Projects/deepim-pytorch/data/YCB_Video/data/0038/'
    num_images = 3466
    is_show = 1

    parameters = np.zeros((6, ), dtype=np.float32)
    parameters[0] = fx
    parameters[1] = fy
    parameters[2] = px
    parameters[3] = py
    parameters[4] = znear
    parameters[5] = zfar

    # set up renderer
    synthesizer = libsynthesizer.Synthesizer(args.cad_name, args.pose_name)
    synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)

    if is_show:
        perm = np.random.permutation(np.arange(num_images))
    else:
        perm = xrange(num_images)

    for i in perm:

        # load meta data
        filename = root + '{:06d}-meta.mat'.format(i+1)
        meta_data = scipy.io.loadmat(filename)

        # prepare data
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        num = poses.shape[2]
        channel = 9
        qt = np.zeros((num, channel), dtype=np.float32)
        for j in xrange(num):
            class_id = int(meta_data['cls_indexes'][j])
            RT = poses[:,:,j]
            print 'class_id', class_id
            print 'RT', RT

            R = RT[:, :3]
            T = RT[:, 3]
            qt[j, 1] = class_id
            qt[j, 2:6] = mat2quat(R)
            qt[j, 6:] = T
        
        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        synthesizer.render_set_python(int(num), int(channel), int(width), int(height), parameters, qt, im_syn)
        im_syn = np.clip(255 * im_syn, 0, 255).astype(np.uint8)

        # show images
        if is_show:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            filename = root + '{:06d}-color.png'.format(i+1)
            im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            im = im[:, :, (2, 1, 0)]
            plt.imshow(im)
            ax.set_title('color') 

            ax = fig.add_subplot(1, 2, 2)
            plt.imshow(im_syn)
            ax.set_title('render') 

            plt.show()
        else:
            # save image
            filename = outdir + '{:06d}-render.png'.format(i+1)
            cv2.imwrite(filename, im_syn)
            print filename
