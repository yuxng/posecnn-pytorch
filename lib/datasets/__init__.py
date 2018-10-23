# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from .imdb import imdb
from .ycb_video import YCBVideo
from .ycb_object import YCBObject
from .linemod import linemod

import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
