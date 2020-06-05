#!/usr/bin/env python3

# --------------------------------------------------------
# FCN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a Fully Convolutional Network (FCN) on image segmentation database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
import pprint
import numpy as np
import sys
import os
import os.path as osp
import cv2

import _init_paths
import datasets
import networks
from fcn.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
from fcn.train import train, train_autoencoder
from datasets.factory import get_dataset
from ycb_renderer import YCBRenderer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a PoseCNN network')
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train',
                        default=40000, type=int)
    parser.add_argument('--startepoch', dest='startepoch',
                        help='the starting epoch',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='solver type',
                        default='sgd', type=str)
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
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
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

    # prepare dataset
    cfg.MODE = 'TRAIN'
    dataset = get_dataset(args.dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, num_workers=0)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    if cfg.INPUT == 'COLOR':
        if cfg.TRAIN.SYN_BACKGROUND_SPECIFIC:
            background_dataset = get_dataset('background_nvidia')
        else:
            background_dataset = get_dataset('background_coco')
    else:
        background_dataset = get_dataset('background_rgbd')
    background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                                    shuffle=True, num_workers=8)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        background_dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    output_dir = get_output_dir(dataset, None)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare networks
    autoencoders = [[] for i in range(len(cfg.TRAIN.CLASSES))]
    for i in range(len(cfg.TRAIN.CLASSES)):
        ind = cfg.TRAIN.CLASSES[i]
        cls = dataset._classes_all[ind]
        filename = args.pretrained.replace('cls', cls)
        autoencoder_data = torch.load(filename)
        autoencoders[i] = networks.__dict__['autoencoder'](1, 128, autoencoder_data).cuda()
        autoencoders[i] = torch.nn.DataParallel(autoencoders[i]).cuda()
        print(filename)

    if torch.cuda.device_count() > 1:
        cfg.TRAIN.GPUNUM = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    cudnn.benchmark = True

    if cfg.TRAIN.SYNTHESIZE:
        print('loading 3D models')
        cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, render_marker=False)
        cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
        cfg.renderer.set_camera_default()
        print(dataset.model_mesh_paths)

    # prepare optimizers
    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    optimizers = [[] for i in range(len(cfg.TRAIN.CLASSES))]
    schedulers = [[] for i in range(len(cfg.TRAIN.CLASSES))]
    optimizers_discriminator = [[] for i in range(len(cfg.TRAIN.CLASSES))]
    schedulers_discriminator = [[] for i in range(len(cfg.TRAIN.CLASSES))]
    for i in range(len(cfg.TRAIN.CLASSES)):
        network = autoencoders[i]

        # autoencoder
        param_groups = [{'params': network.module.bias_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                        {'params': network.module.weight_parameters(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if args.solver == 'adam':
            optimizer = torch.optim.Adam(param_groups, cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
        elif args.solver == 'sgd':
           optimizer = torch.optim.SGD(param_groups, cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, \
            milestones=[m - args.startepoch for m in cfg.TRAIN.MILESTONES], gamma=cfg.TRAIN.GAMMA)

        # discriminator
        param_groups_discriminator = [{'params': network.module.bias_parameters_discriminator(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
                                      {'params': network.module.weight_parameters_discriminator(), 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if args.solver == 'adam':
            optimizer_discriminator = torch.optim.Adam(param_groups_discriminator, cfg.TRAIN.LEARNING_RATE, betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))
        elif args.solver == 'sgd':
           optimizer_discriminator = torch.optim.SGD(param_groups_discriminator, cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(optimizer_discriminator, \
            milestones=[m - args.startepoch for m in cfg.TRAIN.MILESTONES], gamma=cfg.TRAIN.GAMMA)

        optimizers[i] = optimizer
        schedulers[i] = scheduler
        optimizers_discriminator[i] = optimizer_discriminator
        schedulers_discriminator[i] = scheduler_discriminator

    # start training
    cfg.epochs = args.epochs
    for epoch in range(args.startepoch, args.epochs):

        # for each class
        for i in range(len(cfg.TRAIN.CLASSES)):

            schedulers[i].step()
            schedulers_discriminator[i].step()
            dataloader.dataset.cls_target = i
            train_autoencoder(dataloader, background_loader, autoencoders[i], optimizers[i], optimizers_discriminator[i], epoch)

            # save checkpoint
            if (epoch+1) % cfg.TRAIN.SNAPSHOT_EPOCHS == 0 or epoch == args.epochs - 1:
                state = autoencoders[i].module.state_dict()
                infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
                cls = dataloader.dataset.classes[i]
                filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_' + cls + '_epoch_{:d}'.format(epoch+1) + '.checkpoint.pth')
                torch.save(state, os.path.join(output_dir, filename))
                print(filename)

        # update data loader
        # dataset = get_dataset(args.dataset_name)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, num_workers=0)
