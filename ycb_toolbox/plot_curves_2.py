import sys
import os.path as osp
import numpy as np
import scipy.io
from transforms3d.quaternions import mat2quat, quat2mat
from se3 import se3_mul
from pose_error import *
import matplotlib.pyplot as plt

methods = {'Ours RGB', 'Ours RGB-D', 'Ours RGB refine', 'Ours RGB-D refine'}
metrics = {'ADD', 'ADD-S'}

# 0%, 20%, 40%, 60%, 80%, 100% real data
data = [10, 20, 30, 40, 50]

ours_rgb_add = [0.276973, 0.308285, 0.32403, 0.335109, 0.335606]
ours_rgb_adds = [0.482142, 0.520099, 0.540709, 0.551583, 0.553141]
ours_rgb_add_refine = [0.715478, 0.725452, 0.728615, 0.729486, 0.729933]
ours_rgb_adds_refine = [0.890203, 0.893436, 0.895566, 0.89578, 0.895867]

ours_rgbd_add = [0.501155, 0.625493, 0.661104, 0.678174, 0.685989]
ours_rgbd_adds = [0.751968, 0.844966, 0.863904, 0.87299, 0.876182]
ours_rgbd_add_refine = [0.594924, 0.676489, 0.700008, 0.712272, 0.716094]
ours_rgbd_adds_refine = [0.837913, 0.880704, 0.890356, 0.894512, 0.896134]

# plot add
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

plt.plot(data, ours_rgb_add, 'r', linewidth=2)
plt.plot(data, ours_rgb_add_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_add, 'g', linewidth=2)
plt.plot(data, ours_rgbd_add_refine, 'g--', linewidth=2)

#ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine'])

plt.plot(data, ours_rgb_add, 'ro')
plt.plot(data, ours_rgb_add_refine, 'ro')

plt.plot(data, ours_rgbd_add, 'go')
plt.plot(data, ours_rgbd_add_refine, 'go')

plt.xlabel('Number of particles', fontsize=12)
plt.ylabel('ADD', fontsize=12)
ax.set_title('20 YCB objects')

# plot adds
ax = fig.add_subplot(1, 2, 2)

plt.plot(data, ours_rgb_adds, 'r', linewidth=2)
plt.plot(data, ours_rgb_adds_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_adds, 'g', linewidth=2)
plt.plot(data, ours_rgbd_adds_refine, 'g--', linewidth=2)

#ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine'])

plt.plot(data, ours_rgb_adds, 'ro')
plt.plot(data, ours_rgb_adds_refine, 'ro')

plt.plot(data, ours_rgbd_adds, 'go')
plt.plot(data, ours_rgbd_adds_refine, 'go')

plt.xlabel('Number of particles', fontsize=12)
plt.ylabel('ADD-S', fontsize=12)
ax.set_title('20 YCB objects')

plt.show()

