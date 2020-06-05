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


def process_label_image(label_image, class_colors, cls_indexes):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    labels = np.zeros((height, width), dtype=np.int32)

    # label image is in BGR order
    index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
    for i in range(len(class_colors)):
        color = class_colors[i]
        ind = color[0] + 256*color[1] + 256*256*color[2]
        I = np.where(index == ind)
        labels[I[0], I[1]] = cls_indexes[i]
    return labels


if __name__ == '__main__':

    args = parse_args()
    opt = ycb_video()
    root_path = '/capri/YCB_Video_Dataset'
    save_path = '/capri/YCB-real/ycb'
    image_set = 'train_few'
    height = 480
    width = 640
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    is_save = True
    renderer = None

    # load index set
    image_set_file = os.path.join(root_path, 'image_sets', image_set + '.txt')
    image_index = []
    with open(image_set_file) as f:
        for x in f.readlines():
            index = x.rstrip('\n')
            image_index.append(index)

    num_images = len(image_index)
    prev_video_id = None
    count = np.zeros((22, ), dtype=np.int32)
    for i in range(0, num_images, 100):
        # read annotation
        filename = os.path.join(root_path, 'data', image_index[i] + '-meta.mat')
        meta_data = scipy.io.loadmat(filename)
        cls_indexes = meta_data['cls_indexes'].flatten()
        box = meta_data['box']

        pos = image_index[i].find('/')
        video_id = image_index[i][:pos]

        # initialize render
        if prev_video_id is None or video_id != prev_video_id:
            num_classes = len(cls_indexes)
            intrinsic_matrix = meta_data['intrinsic_matrix']
            obj_paths = ['{}/models/{}/textured_simple.obj'.format(root_path, opt.classes[int(cls)]) for cls in cls_indexes]
            texture_paths = ['{}/models/{}/texture_map.png'.format(root_path, opt.classes[int(cls)]) for cls in cls_indexes]
            class_colors = [np.array(opt.class_colors[int(cls)]) for cls in cls_indexes]
            colors = [np.array(opt.class_colors[int(cls)]) / 255.0 for cls in cls_indexes]

            if renderer:
                renderer.clean()
                renderer = None

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

        # read image
        filename = os.path.join(root_path, 'data', image_index[i] + '-color.jpg')
        im = cv2.imread(filename)
        W = im.shape[1]
        H = im.shape[0]

        # read label
        filename = os.path.join(root_path, 'data', image_index[i] + '-label.png')
        im_label_gt = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # prepare data
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))

        # for each object
        num = len(cls_indexes)
        for j in range(num):
            cls = cls_indexes[j]
            cls_name = opt.classes[cls]
            x_min = box[j, 0]
            y_min = box[j, 1]
            x_max = box[j, 2]
            y_max = box[j, 3]

            # Make bbox square
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            x_padding = int((x_max - x_min) * 0.2)
            y_padding = int((y_max - y_min) * 0.2)
            if x_min - x_padding <= 0 or y_min - y_padding <= 0 or x_max + x_padding >= W or y_max + y_padding >= H:
                continue

            # Pad and be careful of boundaries
            x_min = int(max(x_min - x_padding, 0))
            x_max = int(min(x_max + x_padding, W-1))
            y_min = int(max(y_min - y_padding, 0))
            y_max = int(min(y_max + y_padding, H-1))
            if x_max <= x_min or y_max <= y_min:
                continue

            im_new = im.copy()
            I = np.where(im_label_gt != cls)
            im_new[I[0], I[1], :] = 0
            roi = im_new[y_min:y_max, x_min:x_max, :]

            # pose
            RT = poses[:,:,j]
            qt = np.zeros((7, ), dtype=np.float32)
            qt[3:] = mat2quat(RT[:, :3])
            qt[:3] = RT[:, 3]
            poses_all = []
            poses_all.append(qt)

            # rendering
            renderer.set_poses(poses_all)
            renderer.set_light_pos([0, 0, 0])
            renderer.render([j], image_tensor, seg_tensor)
            image_tensor = image_tensor.flip(0)
            seg_tensor = seg_tensor.flip(0)

            im_syn = image_tensor.cpu().numpy()
            im_syn = im_syn[:, :, :3] * 255
            im_syn = np.clip(im_syn, 0, 255)
            im_syn = im_syn.astype(np.uint8)

            im_label = seg_tensor.cpu().numpy()
            im_label = im_label[:, :, (2, 1, 0)] * 255
            im_label = np.round(im_label).astype(np.uint8)
            im_label = np.clip(im_label, 0, 255)
            im_label = process_label_image(im_label, class_colors, cls_indexes)

            # compute occlusion percentage
            non_occluded = np.sum(np.logical_and(im_label_gt > 0, im_label_gt == im_label)).astype(np.float)
            occluded_ratio = 1 - non_occluded / np.sum(im_label > 0).astype(np.float)

            # save roi
            if is_save:
                if occluded_ratio < 0.05:
                    save_dir = os.path.join(save_path, cls_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    filename = os.path.join(save_dir, '%06d.jpg' % count[cls])
                    print('%d:%d, %s' % (i, num_images, filename))
                    # resize roi
                    roi = cv2.resize(roi, (256, 256))
                    cv2.imwrite(filename, roi)
                    count[cls] += 1
            else:
                # show images
                import matplotlib.pyplot as plt
                fig = plt.figure()
                m = 3
                n = 3
                start = 1
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(im[:, :, (2, 1, 0)])
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(im_label_gt)
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(im_syn)
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(im_label)
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(roi[:, :, (2, 1, 0)])
                ax.set_title('occlusion %.4f' % (occluded_ratio))
                plt.show()

        prev_video_id = video_id
