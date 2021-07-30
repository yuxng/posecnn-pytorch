# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
sys.path.insert(0, '/home/yuxiang/Projects/OMG-Planner')

lib_path = osp.join(this_dir, '..', 'ycb_render')
add_path(lib_path)

lib_path = osp.join(this_dir, '..', 'ycb_render', 'robotPose')
add_path(lib_path)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
