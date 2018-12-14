#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2

import _init_paths
from fcn.train_test import test_image
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.device = torch.device('cuda:{:d}'.format(args.gpu_id))
    print 'GPU device {:d}'.format(args.gpu_id)

    # dataset
    cfg.MODE == 'TEST'
    dataset = get_dataset(args.dataset_name)

    # list images
    images = []
    files = os.listdir(args.imgdir)
    for i in range(len(files)):
        filename = os.path.join(args.imgdir, files[i])
        images.append(filename)

    if cfg.TEST.VISUALIZE:
        images = np.random.permutation(images)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[args.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    if cfg.TRAIN.VERTEX_REG and cfg.TEST.POSE_REFINE:
        cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, render_marker=True)
        cfg.renderer.load_objects(dataset.model_mesh_paths_target,
                                  dataset.model_texture_paths_target,
                                  dataset.model_colors_target)
        cfg.renderer.set_camera_default()
        cfg.renderer.set_light_pos([0, 0, 0])
        cfg.renderer.set_light_color([1, 1, 1])
        print dataset.model_mesh_paths_target

    # run network
    for i in range(len(images)):
        im = pad_im(cv2.imread(images[i], cv2.IMREAD_COLOR), 16)
        depth = None
        im_pose, im_label, rois, poses = test_image(network, dataset, im, depth)
