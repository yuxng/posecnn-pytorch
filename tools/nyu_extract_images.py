import os.path as osp
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

root = '/capri/NYUv2'

filename = '/capri/nyu_depth_v2_labeled.mat'
f = h5py.File(filename)
for k, v in f.items():
    if k == 'images':
        images = np.array(v)
        break

print images.shape
for i in range(images.shape[0]):
    im = images[i, :, :, :]
    im = im.transpose((2, 1, 0)).astype(np.uint8)
    im = im[:, :, (2, 1, 0)]

    filename = osp.join(root, '%06d.png' % i)
    cv2.imwrite(filename, im)
    print filename
