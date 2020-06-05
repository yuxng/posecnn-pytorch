#!/usr/bin/env python3

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
import scipy.io
import glob

import _init_paths
from fcn.test_imageset import test_image_cosegmentation
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from fcn.segmentation import Segmentor

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
    parser.add_argument('--pretrained_rrn', dest='pretrained_rrn',
                        help='initialize with pretrained checkpoint RRN',
                        default=None, type=str)
    parser.add_argument('--pretrained_cor', dest='pretrained_cor',
                        help='initialize with pretrained checkpoint correspondences',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--cfg1', dest='cfg_file_1',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--network_cor', dest='network_name_cor',
                        help='name of the network correspondences',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)
    parser.add_argument('--image_a_path', dest='image_a_path', help='path to first image', default=None, type=str)
    parser.add_argument('--image_b_path', dest='image_b_path', help='path to second image', default=None, type=str)

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

    if args.cfg_file_1 is not None:
        cfg_from_file(args.cfg_file_1)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    print('GPU device {:d}'.format(args.gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    # prepre region refinement network
    if args.pretrained_rrn:
        params = {
            # Padding for Region Refinement Network
            'padding_percentage' : 0.25,
            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,
            # Closest Connected Component for IMP module
            'use_closest_connected_component' : True,
        }
        segmentor = Segmentor(params, args.pretrained_rrn)
    else:
        segmentor = None

    if args.pretrained_cor:
        network_data_cor = torch.load(args.pretrained_cor)
        print("=> using pre-trained network '{}'".format(args.pretrained_cor))
        network_cor = networks.__dict__[args.network_name_cor](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data_cor).cuda(device=cfg.device)
        network_cor = torch.nn.DataParallel(network_cor, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        network_cor.eval()
    else:
        network_cor = None

    image_a_path = args.image_a_path.split(' ')
    image_b_path = args.image_b_path.split(' ')

    for i in range(len(image_a_path)):
        if os.path.exists(image_a_path[i]) and os.path.exists(image_b_path[i]):
            # read images
            img_a = pad_im(cv2.imread(image_a_path[i], cv2.IMREAD_COLOR), 16)
            img_b = pad_im(cv2.imread(image_b_path[i], cv2.IMREAD_COLOR), 16)
            print(image_a_path[i], image_b_path[i])

            out_label_a, out_label_b = test_image_cosegmentation(network, dataset, img_a, img_b, segmentor, network_cor)
        else:
            print('files not exist %s, %s' % (image_a_path[i], image_b_path[i]))
