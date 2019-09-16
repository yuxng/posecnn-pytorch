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
import scipy.io
import glob

import _init_paths
from fcn.train_test import test_image
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from utils.blob import pad_im
from fcn.pose_rbpf import PoseRBPF
from sdf.sdf_optimizer import sdf_optimizer

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
    print('GPU device {:d}'.format(args.gpu_id))

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    # list images
    images_color = []
    images_depth = []
    filename = os.path.join(args.imgdir, '*color.png')
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
        images_depth.append(filename.replace('color', 'depth'))

    if cfg.TEST.VISUALIZE:
        index_images = np.random.permutation(len(images_color))
    else:
        index_images = np.range(len(images_color))
        resdir = args.imgdir + '_posecnn_results'
        if not os.path.exists(resdir):
            os.makedirs(resdir)

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

    #'''
    print('loading 3D models')
    cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=args.gpu_id, render_marker=False)
    if cfg.TEST.SYNTHESIZE:
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
    else:
        model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES]
        model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES]
        model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES]
        cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
    cfg.renderer.set_camera_default()
    print(dataset.model_mesh_paths)
    #'''

    # load sdfs
    if cfg.TEST.POSE_REFINE:
        print('loading SDFs')
        cfg.sdf_optimizers = []
        for i in cfg.TEST.CLASSES:
            cfg.sdf_optimizers.append(sdf_optimizer(dataset.model_sdf_paths[i-1]))

    # prepare autoencoder and codebook
    if cfg.TRAIN.VERTEX_REG:
        pose_rbpf = PoseRBPF(dataset)
    else:
        pose_rbpf = None

    # run network
    for i in index_images:
        print(files[i])
        im = pad_im(cv2.imread(images_color[i], cv2.IMREAD_COLOR), 16)
        if osp.exists(images_depth[i]):
            depth = pad_im(cv2.imread(images_depth[i], cv2.IMREAD_UNCHANGED), 16)
            depth = depth.astype('float') / 1000.0
        else:
            depth = None
        im_pose, labels, rois, poses = test_image(network, pose_rbpf, dataset, im, depth)

        # save result
        if not cfg.TEST.VISUALIZE:
            result = {'labels': labels, 'rois': rois, 'poses': poses, 'intrinsic_matrix': dataset._intrinsic_matrix}
            head, tail = os.path.split(images_color[i])
            filename = os.path.join(resdir, tail + '.mat')
            print(images_color[i], filename)
            scipy.io.savemat(filename, result, do_compression=True)
