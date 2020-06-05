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
import torch.nn.functional as F

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob

import _init_paths
import datasets
import networks
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
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
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--save_name', dest='save_name',
                        help='filename of the codebook',
                        default='codebook_ycb_embeddings', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
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

    # list objects
    object_paths, object_nums, object_names = dataset.list_objects(args.imgdir)

    # for each object
    num_obj = len(object_paths)
    dim = cfg.TRAIN.NUM_UNITS
    codes = np.zeros((0, dim), dtype=np.float32)
    object_indexes = np.zeros((0, ), dtype=np.int32)
    centers = np.zeros((num_obj, dim), dtype=np.float32)
    filenames = []
    for i in range(num_obj):
        path = object_paths[i]
        num = object_nums[i]
        features = torch.zeros((num, dim), dtype=torch.float32, device=cfg.device)
        # for each image
        for j in range(num):
            filename = os.path.join(path, '%06d.jpg' % (j))
            filenames.append(filename)
            print(filename)
            im = cv2.imread(filename)
            im_blob = dataset.transform_image(im).cuda().unsqueeze(0)
            f = network(im_blob).detach()
            features[j] = f

        codes = np.concatenate((codes, features.cpu().numpy()), axis=0)
        obj_indexes = i * np.ones((num, ), dtype=np.int32)
        object_indexes = np.concatenate((object_indexes, obj_indexes), axis=0)

        # compute mean
        center = torch.sum(features, dim=0)
        center = F.normalize(center, p=2, dim=0)
        centers[i, :] = center.cpu().numpy()

    # save embeddings
    codebook = {'codes': codes, 'centers': centers, 'object_indexes': object_indexes, 'object_names': object_names, 'filenames': filenames}
    output_dir = os.path.join(datasets.ROOT_DIR, 'data', 'codebooks')
    filename = os.path.join(output_dir, args.save_name + '.mat')
    print('save codebook to %s' % (filename))
    scipy.io.savemat(filename, codebook, do_compression=True)
