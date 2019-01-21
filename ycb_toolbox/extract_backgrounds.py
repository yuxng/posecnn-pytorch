import sys
import os.path as osp
import numpy as np
import scipy.io
import cv2
from ycb_globals import ycb_video

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
opt = ycb_video()

# read keyframe index
filename = osp.join(root_path, 'train.txt')
keyframes = []
with open(filename) as f:
    for x in f.readlines():
        index = x.rstrip('\n')
        keyframes.append(index)

# for each image
count = 0
is_show = 0
for i in range(0, len(keyframes), 100):

    # load image
    filename = osp.join(root_path, 'data', '%s-color.png' % (keyframes[i]))
    print(keyframes[i])
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    # load label
    filename = osp.join(root_path, 'data', '%s-label.png' % (keyframes[i]))
    label = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    mask = (label != 0).astype(np.uint8)
    kernel = np.ones((20, 20), np.uint8) 
    mask = cv2.dilate(mask, kernel, iterations=1) 
    dst = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)
    
    if is_show:
        # show images
        import matplotlib.pyplot as plt
        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(im[:, :, (2, 1, 0)])
        ax.set_title('color')

        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(label)
        ax.set_title('label')

        ax = fig.add_subplot(2, 2, 4)
        plt.imshow(mask)
        ax.set_title('mask')

        ax = fig.add_subplot(2, 2, 3)
        plt.imshow(dst[:, :, (2, 1, 0)])
        ax.set_title('output')

        plt.show()
    else:
        filename = osp.join('ycb_backgrounds', '%06d.png' % (count))
        cv2.imwrite(filename, dst)
        count += 1
