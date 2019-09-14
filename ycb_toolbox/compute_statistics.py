import os
import os.path as osp
import numpy as np
import glob
import scipy.io
from ycb_globals import ycb_video

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..', 'data', 'YCB_Self_Supervision', 'data')
opt = ycb_video()
classes = opt.classes

num_scenes = 0
num_images = 0
num_classes = len(classes)
count = np.zeros((num_classes, ), dtype=np.int32)

# list subdirs
subdirs = os.listdir(root_path)
for i in range(len(subdirs)):
    subdir = subdirs[i]
    path_sub = osp.join(root_path, subdir)

    # list subsubdirs
    subsubdirs = os.listdir(path_sub)
    for j in range(len(subsubdirs)):
        subsubdir = subsubdirs[j]
        folder = osp.join(root_path, subdir, subsubdir)
        if osp.isdir(folder):
            print(folder)
            num_scenes += 1

            filename = osp.join(folder, '*.mat')
            files = glob.glob(filename)
            for k in range(len(files)):
                filename = files[k]
                num_images += 1
                # load the annotation to see if the target object is in the image
                meta_data = scipy.io.loadmat(filename)
                cls_indexes = meta_data['cls_indexes'].flatten()
                count[cls_indexes - 1] += 1

print('num of scenes: %d' % (num_scenes))
print('num of images: %d' % (num_images))
print('num of objects: %d' % (np.sum(count)))
print('avg objects per image: %f' % (float(np.sum(count)) / float(num_images)))
print('==============================')
for i in range(num_classes):
    if count[i] > 0:
        print('%s: %d' % (classes[i], count[i]))
print('==============================')
