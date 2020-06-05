#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
import torch
import argparse
import os, sys
import cv2
import numpy as np
from pathlib import Path
from shutil import copyfile, copytree

if __name__ == '__main__':

    dst_dir = '/capri/ShapeNetCore-nomat'
    src_dir = '/capri/ShapeNetCore.v2'

    shape_index_path = (Path(dst_dir) / 'paths.txt')
    with shape_index_path.open('r') as f:
        paths = [p.strip() for p in f.readlines()]

    for i in range(len(paths)):
        name, tail = os.path.split(paths[i])
        print(name)
        src_file = os.path.join(src_dir, name, 'model_normalized.mtl')
        dst_file = os.path.join(dst_dir, name, 'uv_unwrapped.mtl')
        copyfile(src_file, dst_file)

        src_img_dir = os.path.join(src_dir, name.replace('models', 'images'))
        if os.path.exists(src_img_dir):
            dst_img_dir = os.path.join(dst_dir, name.replace('models', 'images'))
            images = os.listdir(src_img_dir)
            for file_name in images:
                full_file_name = os.path.join(src_img_dir, file_name)
                if os.path.isfile(full_file_name):
                    dest = os.path.join(dst_img_dir, file_name)
                    copyfile(full_file_name, dest)

        src_img_dir = os.path.join(src_dir, name, 'untitled')
        if os.path.exists(src_img_dir):
            dst_img_dir = os.path.join(dst_dir, name, 'untitled')
            images = os.listdir(src_img_dir)
            for file_name in images:
                full_file_name = os.path.join(src_img_dir, file_name)
                if os.path.isfile(full_file_name):
                    dest = os.path.join(dst_img_dir, file_name)
                    copyfile(full_file_name, dest)
