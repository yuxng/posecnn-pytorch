#!/usr/bin/env python

import torch
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np

import rospy
import glob
import copy
import _init_paths
from datasets.factory import get_dataset
from mask_listener import ImageListener
from fcn.config import cfg, cfg_from_file

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Start computing depth masks')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

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

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(cfg.gpu_id))
    cfg.instance_id = args.instance_id

    # image listener
    listener = ImageListener(dataset)
    while not rospy.is_shutdown():
        if listener.input_depth is not None:
            listener.process_data()
