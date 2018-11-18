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
from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
from utils.se3 import *
from ycb_renderer import YCBRenderer
from ycb_globals import ycb_video

if __name__ == '__main__':

    opt = ycb_video()
    root = '/capri/YCB_Video_Dataset'
    height = 480
    width = 640

    filename = os.path.join(root, 'data', '0000', '000001-meta.mat')
    metadata = scipy.io.loadmat(filename)
    intrinsic_matrix = metadata['intrinsic_matrix']
    cls_indexes = [2]
    num_classes = len(cls_indexes)

    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(root, opt.classes[int(cls)-1]) for cls in cls_indexes]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(root, opt.classes[int(cls)-1]) for cls in cls_indexes]
    colors = [np.array(opt.class_colors[int(cls)-1]) / 255.0 for cls in cls_indexes]

    renderer = YCBRenderer(width=width, height=height, render_marker=True)
    renderer.load_objects(obj_paths, texture_paths, colors)

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 100.0
    znear = 0.25
    z = 8.0
    renderer.set_camera_default()
    renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
    renderer.set_light_pos([0, 1, 1])
    renderer.instances = [0] * 100

    # allocentric
    poses = []
    for i in np.arange(-2, 2, 0.4):
        for j in np.arange(-2, 2, 0.4):
            quat = [1, 0, 0, 0]
            poses.append(np.array([i, j, z] + list(quat)))

    renderer.set_allocentric_poses(poses)
    frame = renderer.render([0] * len(poses))
    im_syn0 = frame[0][:, :, :3] * 255
    im_syn0 = np.clip(im_syn0, 0, 255)
    im_syn0 = im_syn0.astype(np.uint8)

    # egocentric
    poses_new = []
    poses = renderer.get_poses()
    for i in range(len(poses)):
        T = poses[i][:3]
        quat = egocentric2allocentric(poses[i][3:], T)
        poses_new.append(np.array(list(T) + list(quat)))

    renderer.set_allocentric_poses(poses_new)
    frame = renderer.render([0] * len(poses))
    im_syn1 = frame[0][:, :, :3] * 255
    im_syn1 = np.clip(im_syn1, 0, 255)
    im_syn1 = im_syn1.astype(np.uint8)

    # show images
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(im_syn0[:,:,(2, 1, 0)])
    ax.set_title('allocentric')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(im_syn1[:,:,(2, 1, 0)])
    ax.set_title('egocentric to allocentric')

    plt.show()
