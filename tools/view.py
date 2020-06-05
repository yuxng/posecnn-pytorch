#!/usr/bin/env python3
import sys, os
import argparse
import pyglet
import glob
import torch
import numpy as np
import _init_paths
import window
from datasets.image_loader import ImageLoader
from fcn.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(
        description='View point cloud and ground-truth hand & object poses in 3D.'
    )
    parser.add_argument('--name',
                        help="Name of the sequence",
                        default=None,
                        type=str)
    parser.add_argument('--no-preload', action='store_true', default=False)
    parser.add_argument('--use-cache', action='store_true', default=False)
    parser.add_argument('--device',
                        help='Device for data loader computation',
                        default='cuda:0',
                        type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default='data/Images', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
    images_color.sort()

    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    # image loader
    device = 'cuda:{:d}'.format(cfg.gpu_id)
    loader = ImageLoader(images_color, images_depth, device)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        if cfg.TEST.SCALES_BASE[0] != 1:
            scale = cfg.TEST.SCALES_BASE[0]
            K[0, 0] *= scale
            K[0, 2] *= scale
            K[1, 1] *= scale
            K[1, 2] *= scale
        loader._intrinsic_matrix = K
        loader._master_intrinsics = K
        print(loader._intrinsic_matrix)

    w = window.Window(loader)

    def run(dt):
      w.update()

    pyglet.clock.schedule(run)
    pyglet.app.run()
