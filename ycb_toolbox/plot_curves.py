import sys
import os.path as osp
import numpy as np
import scipy.io
from transforms3d.quaternions import mat2quat, quat2mat
from se3 import se3_mul
from pose_error import *
import matplotlib.pyplot as plt

methods = {'PoseCNN RGB', 'Ours RGB', 'Ours RGB-D', 'Ours RGB refine', 'Ours RGB-D refine'}
metrics = {'ADD', 'ADD-S'}

# 0%, 20%, 40%, 60%, 80%, 100% real data
data = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

ours_rgb_add = [0.168984, 0.282818, 0.292411, 0.302877, 0.301693, 0.308285]
ours_rgb_adds = [0.339209, 0.497321, 0.508587, 0.514795, 0.512697, 0.520099]
ours_rgb_add_refine = [0.488433, 0.708435, 0.708519, 0.716873, 0.717745, 0.725452]
ours_rgb_adds_refine = [0.6895, 0.883772, 0.886891, 0.889974, 0.890648, 0.893436]

ours_rgbd_add = [0.422938, 0.606861, 0.609502, 0.619793, 0.621018, 0.625493]
ours_rgbd_adds = [0.666392, 0.834124, 0.83733, 0.841788, 0.841845, 0.844966]
ours_rgbd_add_refine = [0.465191, 0.655686, 0.659282, 0.669756, 0.673267, 0.676489]
ours_rgbd_adds_refine = [0.707161, 0.869362, 0.873828, 0.876325, 0.877726, 0.880704]

posecnn_rgb_add = [0.131786, 0.306563, 0.330913, 0.346786, 0.361857, 0.374973]
posecnn_rgb_adds = [0.365851, 0.571387, 0.596357, 0.605353, 0.618053, 0.627665]
posecnn_rgb_add_refine = [0.329861, 0.586816, 0.601019, 0.625298, 0.635857, 0.650099]
posecnn_rgb_adds_refine = [0.645043, 0.854424, 0.861451, 0.868557, 0.869985, 0.875049]

# plot add
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

plt.plot(data, ours_rgb_add, 'r', linewidth=2)
plt.plot(data, ours_rgb_add_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_add, 'g', linewidth=2)
plt.plot(data, ours_rgbd_add_refine, 'g--', linewidth=2)

plt.plot(data, posecnn_rgb_add, 'b', linewidth=2)
plt.plot(data, posecnn_rgb_add_refine, 'b--', linewidth=2)

#ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine', 'PoseCNN RGB [1]', 'PoseCNN RGB refine [1]'])

plt.plot(data, ours_rgb_add, 'ro')
plt.plot(data, ours_rgb_add_refine, 'ro')

plt.plot(data, ours_rgbd_add, 'go')
plt.plot(data, ours_rgbd_add_refine, 'go')

plt.plot(data, posecnn_rgb_add, 'bo')
plt.plot(data, posecnn_rgb_add_refine, 'bo')

plt.xlabel('Percentage of real training data', fontsize=16)
plt.ylabel('ADD', fontsize=16)
ax.set_title('20 YCB objects', fontsize=16)

# plot adds
ax = fig.add_subplot(1, 2, 2)

plt.plot(data, ours_rgb_adds, 'r', linewidth=2)
plt.plot(data, ours_rgb_adds_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_adds, 'g', linewidth=2)
plt.plot(data, ours_rgbd_adds_refine, 'g--', linewidth=2)

plt.plot(data, posecnn_rgb_adds, 'b', linewidth=2)
plt.plot(data, posecnn_rgb_adds_refine, 'b--', linewidth=2)

#ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine', 'PoseCNN RGB [1]', 'PoseCNN RGB refine [1]'])

plt.plot(data, ours_rgb_adds, 'ro')
plt.plot(data, ours_rgb_adds_refine, 'ro')

plt.plot(data, ours_rgbd_adds, 'go')
plt.plot(data, ours_rgbd_adds_refine, 'go')

plt.plot(data, posecnn_rgb_adds, 'bo')
plt.plot(data, posecnn_rgb_adds_refine, 'bo')

plt.xlabel('Percentage of real training data', fontsize=16)
plt.ylabel('ADD-S', fontsize=16)
ax.set_title('20 YCB objects', fontsize=16)

plt.show()

