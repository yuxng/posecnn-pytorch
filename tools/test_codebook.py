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
import matplotlib.pyplot as plt
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
    parser.add_argument('--codebook', dest='codebook_name',
                        help='name of the codebook file',
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

    # load codebook
    codebook = scipy.io.loadmat(args.codebook_name)
    filenames = codebook['filenames']
    object_indexes = codebook['object_indexes'].flatten()
    codes = torch.from_numpy(codebook['codes']).cuda()
    num = codes.shape[0]
    metric = cfg.TRAIN.EMBEDDING_METRIC
    K = 5
    visualize = False

    # for each image
    index = np.random.permutation(num)
    count = 0
    error = 0
    for i in index:
        print(count)
        # compare code to codebook
        obj_index = object_indexes[i]
        code = codes[i, :].unsqueeze(0)

        if metric == 'euclidean':
            norm_degree = 2
            distance = (code - codes).norm(norm_degree, 1)
        elif metric == 'cosine':
            distance = 0.5 * (1 - (code * codes).sum(dim=1))

        distance_sorted, indexes = torch.sort(distance)

        # compute error
        for j in range(K):
            ind = indexes[j + 1]
            if object_indexes[ind] != obj_index:
                error += 1

        # visualization
        if visualize:
            if count % 5 == 0:
                fig = plt.figure()
                m = 5
                n = 1 + K
                start = 1
        count += 1
        if visualize:
            ax = fig.add_subplot(m, n, start)
            start += 1
            filename = filenames[i].rstrip()
            im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax.set_title('query')
            plt.axis('off')

            for j in range(K):
                ind = indexes[j + 1]
                filename = filenames[ind].rstrip()
                im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                ax = fig.add_subplot(m, n, start)
                start += 1
                plt.imshow(im[:, :, (2, 1, 0)])
                ax.set_title('%4f' % distance_sorted[j+1])
                plt.axis('off')

            if count % 5 == 0:
                plt.show()

    print('error count %d' % (error))
    print('error rate %.2f' % (100 * float(error) / float(K * num)))
