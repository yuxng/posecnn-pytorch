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
from transforms3d.quaternions import mat2quat, quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.se3 import *
from ycb_renderer import YCBRenderer
from ycb_globals import ycb_video

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
    root = '/capri/YCB_Video_Dataset'
    seq_id = '0038'
    height = 480
    width = 640
    start = 0
    is_save = False

    # load the first mat file
    filename = os.path.join(root, 'data', seq_id, '000001-meta.mat')
    metadata = scipy.io.loadmat(filename)
    cls_indexes = metadata['cls_indexes'].flatten()
    num_classes = len(cls_indexes)
    intrinsic_matrix = metadata['intrinsic_matrix']

    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(root, opt.classes[int(cls)]) for cls in cls_indexes]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(root, opt.classes[int(cls)]) for cls in cls_indexes]
    colors = [np.array(opt.class_colors[int(cls)]) / 255.0 for cls in cls_indexes]

    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    renderer.load_objects(obj_paths, texture_paths, colors)

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 6.0
    znear = 0.01

    renderer.set_camera_default()
    renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    num_images = min(500, opt.nums[int(seq_id)])
    if is_save:
        perm = range(num_images)
    else:
        perm = np.random.permutation(np.arange(num_images))
    for i in perm:

        # load meta data
        filename = os.path.join(root, 'data', seq_id, '{:06d}-meta.mat'.format(i+1))
        meta_data = scipy.io.loadmat(filename)
        cls_indexes = np.arange(num_classes)

        # prepare data
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))
        num = poses.shape[2]
        
        poses_all = []
        for j in range(num):
            RT = poses[:,:,j]
            qt = np.zeros((7, ), dtype=np.float32)
            qt[3:] = mat2quat(RT[:, :3])
            qt[:3] = RT[:, 3]
            poses_all.append(qt)

        renderer.set_poses(poses_all)
        renderer.set_light_pos([0, 0, 0])

        renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        seg_tensor = seg_tensor.flip(0)
        frame = [image_tensor.cpu().numpy(), seg_tensor.cpu().numpy()]

        im_syn = frame[0][:, :, :3] * 255
        im_syn = np.clip(im_syn, 0, 255)
        im_syn = im_syn.astype(np.uint8)

        im_label = frame[1][:, :, :3] * 255
        im_label = np.clip(im_label, 0, 255)
        im_label = im_label.astype(np.uint8)

        filename = os.path.join(root, 'data', seq_id, '{:06d}-color.jpg'.format(i+1))
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        filename = os.path.join(root, 'data', seq_id, '{:06d}-depth.png'.format(i+1))
        depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # show images
        if is_save:
            # color
            filename = os.path.join('data', 'color', '{:06d}.png'.format(start+i+1))
            print(filename)
            cv2.imwrite(filename, im)
            # depth
            colors = plt.cm.viridis(plt.Normalize(np.min(depth), np.max(depth))(depth)) * 255
            colors = np.clip(colors, 0, 255)
            colors = colors.astype(np.uint8)
            filename = os.path.join('data', 'depth', '{:06d}.png'.format(start+i+1))
            cv2.imwrite(filename, colors)
            # syn
            filename = os.path.join('data', 'syn', '{:06d}.png'.format(start+i+1))
            cv2.imwrite(filename, im_syn[:, :, (2, 1, 0)])
            # label
            filename = os.path.join('data', 'label', '{:06d}.png'.format(start+i+1))
            cv2.imwrite(filename, im_label[:, :, (2, 1, 0)])
        else:
            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            im = im[:, :, (2, 1, 0)]
            plt.imshow(im)
            ax.set_title('color')
            ax = fig.add_subplot(2, 2, 2)
            plt.imshow(depth)
            ax.set_title('depth')
            ax = fig.add_subplot(2, 2, 3)
            plt.imshow(im_syn)
            ax.set_title('render')
            ax = fig.add_subplot(2, 2, 4)
            plt.imshow(im_label)
            ax.set_title('label')
            plt.show()
