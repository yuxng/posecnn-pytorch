import sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

methods = {'Ours RGB', 'Ours RGB-D', 'Ours RGB refine', 'Ours RGB-D refine'}
metrics = {'ADD', 'ADD-S'}

# 0%, 20%, 40%, 60%, 80%, 100% real data
data = [0, 1, 2, 3, 4]

ours_rgb_add = [0.314572, 0.319033, 0.308285, 0.304265, 0.299757]
ours_rgb_adds = [0.527191, 0.537257, 0.520099, 0.518407, 0.510379]
ours_rgb_add_refine = [0.724412, 0.727307, 0.725452, 0.72176, 0.717528]
ours_rgb_adds_refine = [0.892981, 0.894014, 0.893436, 0.891633, 0.888884]

ours_rgbd_add = [0.590865, 0.638689, 0.625493, 0.610817, 0.597524]
ours_rgbd_adds = [0.810613, 0.848523, 0.844966, 0.833265, 0.829131]
ours_rgbd_add_refine = [0.694508, 0.692238, 0.676489, 0.660738, 0.645049]
ours_rgbd_adds_refine = [0.888619, 0.888246, 0.880704, 0.869567, 0.863531]

# plot add
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

plt.plot(data, ours_rgb_add, 'r', linewidth=2)
plt.plot(data, ours_rgb_add_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_add, 'g', linewidth=2)
plt.plot(data, ours_rgbd_add_refine, 'g--', linewidth=2)

ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine'])

plt.plot(data, ours_rgb_add, 'ro')
plt.plot(data, ours_rgb_add_refine, 'ro')

plt.plot(data, ours_rgbd_add, 'go')
plt.plot(data, ours_rgbd_add_refine, 'go')

plt.xlabel('Number of filtering steps', fontsize=12)
plt.ylabel('ADD', fontsize=12)
ax.set_title('20 YCB objects')

# plot adds
ax = fig.add_subplot(1, 2, 2)

plt.plot(data, ours_rgb_adds, 'r', linewidth=2)
plt.plot(data, ours_rgb_adds_refine, 'r--', linewidth=2)

plt.plot(data, ours_rgbd_adds, 'g', linewidth=2)
plt.plot(data, ours_rgbd_adds_refine, 'g--', linewidth=2)

ax.legend(['Ours RGB', 'Ours RGB refine', 'Ours RGB-D', 'Ours RGB-D refine'])

plt.plot(data, ours_rgb_adds, 'ro')
plt.plot(data, ours_rgb_adds_refine, 'ro')

plt.plot(data, ours_rgbd_adds, 'go')
plt.plot(data, ours_rgbd_adds_refine, 'go')

plt.xlabel('Number of filtering steps', fontsize=12)
plt.ylabel('ADD-S', fontsize=12)
ax.set_title('20 YCB objects')

plt.show()

