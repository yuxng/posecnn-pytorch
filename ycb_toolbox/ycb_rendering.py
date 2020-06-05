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
import torch
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from transforms3d.euler import euler2quat, quat2euler
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
import math
from utils.se3 import *
from ycb_renderer import YCBRenderer
from ycb_globals import ycb_video

OBJ_DEFAULT_POSE = torch.tensor((
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
))

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

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    opt = ycb_video()
    root = '../data'
    height = 128
    width = 128
    cls_index = 40
    cls_name = opt.classes[cls_index]
    intrinsic_matrix = np.array([[500, 0, 64],
                                 [0, 500, 64],
                                 [0, 0, 1]])

    obj_paths = ['{}/models/{}/textured_simple.obj'.format(root, cls_name)]
    texture_paths = ['']
    colors = [np.array(opt.class_colors[cls_index]) / 255.0]

    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    renderer.load_objects(obj_paths, texture_paths, colors)

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.01

    renderer.set_camera_default()
    renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    # load poses
    filename = '{}/codebooks/codebook_ycb_encoder_test_{}.pth'.format(root, cls_name)
    data = torch.load(filename)
    poses = data[1][0, :, :].cpu().numpy()

    num_images = poses.shape[0]
    perm = np.random.permutation(np.arange(num_images))
    perm = range(0, num_images)
    count = 0
    for i in perm:
        
        poses_all = []
        qt = poses[i, :]

        RT = np.zeros((3, 4), dtype=np.float32)
        RT[:3, :3] = quat2mat(qt[3:])
        RT[:3, 3] = qt[:3]
        y = np.matmul(intrinsic_matrix, np.matmul(RT, np.array([0, 1, 0, 1])))
        x1 = y[0] / y[2]
        y1 = y[1] / y[2]

        if y1 > 64:
            continue
        else:
            count += 1

        poses_all.append(qt)
        renderer.set_poses(poses_all)
        renderer.set_light_pos([0, 0, 0])

        renderer.render([0], image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        frame = [image_tensor.cpu().numpy(), seg_tensor.cpu().numpy()]

        im_syn = frame[0][:, :, :3] * 255
        im_syn = np.clip(im_syn, 0, 255)
        im_syn = im_syn.astype(np.uint8)

        im_label = frame[1][:, :, :3] * 255
        im_label = np.clip(im_label, 0, 255)
        im_label = im_label.astype(np.uint8)

        # show images
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(im_syn)
        plt.plot(x1, y1, 'bo')
        ax.set_title('render')

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(im_label)
        ax.set_title('label')

        plt.show()
    print(count)
