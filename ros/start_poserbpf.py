#!/usr/bin/env python

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

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
from ycb_renderer import YCBRenderer
from poserbpf_listener import ImageListener
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from fcn.pose_rbpf import PoseRBPF
from sdf.sdf_optimizer import sdf_optimizer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Start PoseRBPF ROS Node with Multiple Object Models')
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--instance', dest='instance_id', help='PoseCNN instance id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--codebook', dest='codebook',
                        help='initialize with codebook',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
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
    parser.add_argument('--target_obj', dest='target_object',
                        help='target object to track',
                        default='003_cracker_box',
                        type=str)
    parser.add_argument('--gen_data', dest='gen_data',
                        help='generate training data',
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

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(cfg.gpu_id))
    cfg.instance_id = args.instance_id

    # dataset
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)

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
    pose_rbpf = PoseRBPF(dataset, args.pretrained, args.codebook)

    # image listener
    listener = ImageListener(pose_rbpf, args.gen_data)
    while not rospy.is_shutdown():
        if listener.input_rgb is not None:
            listener.process_data()
