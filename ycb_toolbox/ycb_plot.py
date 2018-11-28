import sys
import os.path as osp
import numpy as np
import scipy.io
import json
from transforms3d.quaternions import mat2quat, quat2mat
from ycb_globals import ycb_video
from se3 import se3_mul
from pose_error import *
import matplotlib.pyplot as plt

def VOCap(rec, prec):
    index = np.where(np.isfinite(rec))[0]
    rec = rec[index]
    prec = prec[index]
    if len(rec) == 0 or len(prec) == 0:
        ap = 0
    else:
        mrec = np.insert(rec, 0, 0)
        mrec = np.append(mrec, 0.1)
        mpre = np.insert(prec, 0, 0)
        mpre = np.append(mpre, prec[-1])
        for i in range(1, len(mpre)):
            mpre[i] = max(mpre[i], mpre[i-1])
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum(np.multiply(mrec[i] - mrec[i-1], mpre[i])) * 10
    return ap

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Video')
opt = ycb_video()

color = ['r', 'b', 'y', 'g', 'c']
leng = ['ISER', 'drfat_yu_single', 'drfat_yu_single_30', 'dr_yu_single', 'PoseCNN_drfat']
num = len(leng)
aps = np.zeros((num, ), dtype=np.float32)

# load results
mat = scipy.io.loadmat('results_comparison_all.mat')
distances_sys = mat['distances_sys']
distances_non = mat['distances_non']
rotations = mat['errors_rotation']
translations = mat['errors_translation']
cls_ids = mat['results_cls_id'].flatten()

index_plot = [0, 1, 2, 3, 4]
max_distance = 0.1
show_classes = [1, 2, 3, 4, 8]

# for each class
for k in show_classes:
    fig = plt.figure()
    index = np.where(cls_ids == k + 1)[0]

    if len(index) == 0:
        continue
    print('%s: %d objects' % (opt.classes[k], len(index)))

    # distance symmetry
    ax = fig.add_subplot(2, 4, 1)
    lengs = []
    for i in index_plot:
        D = distances_sys[index, i]
        ind = np.where(D > max_distance)[0]
        D[ind] = np.inf
        d = np.sort(D)
        n = len(d)
        accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
        plt.plot(d, accuracy, color[i], linewidth=2)
        aps[i] = VOCap(d, accuracy)
        lengs.append('%s (%.2f)' % (leng[i], aps[i] * 100))
        print('%s, %s: %d objects missed' % (opt.classes[k], leng[i], np.sum(np.isinf(D))))

    ax.legend(lengs)
    plt.xlabel('Average distance threshold in meter (symmetry)')
    plt.ylabel('accuracy')
    ax.set_title(opt.classes[k])

    # distance non-symmetry
    ax = fig.add_subplot(2, 4, 2)
    lengs = []
    for i in index_plot:
        D = distances_non[index, i]
        ind = np.where(D > max_distance)[0]
        D[ind] = np.inf
        d = np.sort(D)
        n = len(d)
        accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
        plt.plot(d, accuracy, color[i], linewidth=2)
        aps[i] = VOCap(d, accuracy)
        lengs.append('%s (%.2f)' % (leng[i], aps[i] * 100))
        print('%s, %s: %d objects missed' % (opt.classes[k], leng[i], np.sum(np.isinf(D))))

    ax.legend(lengs)
    plt.xlabel('Average distance threshold in meter (non-symmetry)')
    plt.ylabel('accuracy')
    ax.set_title(opt.classes[k])

    # translation
    ax = fig.add_subplot(2, 4, 3)
    lengs = []
    for i in index_plot:
        D = translations[index, i]
        ind = np.where(D > max_distance)[0]
        D[ind] = np.inf
        d = np.sort(D)
        n = len(d)
        accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
        plt.plot(d, accuracy, color[i], linewidth=2)
        aps[i] = VOCap(d, accuracy)
        lengs.append('%s (%.2f)' % (leng[i], aps[i] * 100))
        print('%s, %s: %d objects missed' % (opt.classes[k], leng[i], np.sum(np.isinf(D))))

    ax.legend(lengs)
    plt.xlabel('Translation threshold in meter')
    plt.ylabel('accuracy')
    ax.set_title(opt.classes[k])

    # rotation histogram
    count = 4
    for i in index_plot:
        ax = fig.add_subplot(2, 4, count)
        D = rotations[index, i]
        ind = np.where(np.isfinite(D))[0]
        D = D[ind]
        ax.hist(D, bins=range(0, 190, 10), range=(0, 180))
        plt.xlabel('Rotation angle error')
        plt.ylabel('count')
        ax.set_title(leng[i])
        count += 1

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    plt.show()
