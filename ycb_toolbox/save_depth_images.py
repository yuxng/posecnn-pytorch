import os
import os.path as osp
import glob
import cv2
import numpy as np


folder = 'data_self_supervision'
filename = os.path.join(folder, '*_depth.png')
files = glob.glob(filename)

for i in range(len(files)):
    filename = files[i]
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    im = im.astype(np.float32)
    im /= 1000.0
    im = np.clip(im, 0, 1)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(im)

    filename = filename[:-4] + '_new.png'
    plt.savefig(filename)
    #plt.show()

