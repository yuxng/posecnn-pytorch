import sys
import os
import os.path as osp
import numpy as np
import scipy.io
import json
import cv2
from transforms3d.quaternions import mat2quat, quat2mat
from ycb_globals import ycb_video
from pose_error import *

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
opt = ycb_video()

# read keyframe index
filename = osp.join(root_path, 'trainval.txt')
keyframes = []
with open(filename) as f:
    for x in f.readlines():
        index = x.rstrip('\n')
        keyframes.append(index)

num = len(keyframes)
for i in range(num):
    name = keyframes[i]
    filename = osp.join(root_path, 'data', name + '-color.png')
    if osp.exists(filename):
        im = cv2.imread(filename)
        os.remove(filename)
        filename = osp.join(root_path, 'data', name + '-color.jpg')
        cv2.imwrite(filename, im)
    print(filename)
