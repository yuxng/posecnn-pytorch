import os
import os.path as osp
import numpy as np
import glob
import scipy.io
from ycb_globals import ycb_video

this_dir = os.getcwd()
root_path = osp.join(this_dir, '..', 'data', 'YCB_Self_Supervision', 'data')
opt = ycb_video()
classes = opt.classes

num_scenes = 0
num_images = 0
num_classes = len(classes)

num_scenes_test = 0
num_images_test = 0
count_test = np.zeros((num_classes, ), dtype=np.int32)

percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
num = len(percentages)
num_scenes_train = np.zeros((num, ), dtype=np.int32)
num_images_train = np.zeros((num, ), dtype=np.int32)
count_train = np.zeros((num_classes, num), dtype=np.int32)

scenes_train = [[] for i in range(num)]
scenes_test = []
scenes_all = []

# list subdirs
subdirs = os.listdir(root_path)
for i in range(len(subdirs)):
    subdir = subdirs[i]
    path_sub = osp.join(root_path, subdir)

    # list subsubdirs
    subsubdirs = [o for o in os.listdir(path_sub) if osp.isdir(osp.join(path_sub, o))]
    length = len(subsubdirs)

    # perturb
    per = np.random.permutation(length)
    subsubdirs = [subsubdirs[i] for i in per]

    for j in range(length):
        subsubdir = subsubdirs[j]
        folder = osp.join(subdir, subsubdir) + '\n'
        print(folder)
        scenes_all.append(folder)
        num_scenes += 1

        if j < length / 2:
            scenes_test.append(folder)
            is_train = 0
            num_scenes_test += 1
        else:
            if length == 1:
                ind = 1
            else:
                ind = float(j - length / 2) / float(length / 2)

            for k in range(num):
                if ind <= percentages[k]:
                    num_scenes_train[k] += 1
                    scenes_train[k].append(folder)
            is_train = 1


        folder = osp.join(root_path, subdir, subsubdir)
        filename = osp.join(folder, '*.mat')
        files = glob.glob(filename)
        for k in range(len(files)):
            filename = files[k]
            num_images += 1
            # load the annotation to see if the target object is in the image
            meta_data = scipy.io.loadmat(filename)
            cls_indexes = meta_data['cls_indexes'].flatten()

            if is_train:
                for k in range(num):
                    if ind <= percentages[k]:
                        count_train[cls_indexes - 1, k] += 1
                        num_images_train[k] += 1
            else:
                count_test[cls_indexes - 1] += 1
                num_images_test += 1

print('num of scenes: %d' % (num_scenes))
print('num of images: %d' % (num_images))

for k in range(num):
    print('=============training %.2f=================' % (percentages[k]))
    print('num of scenes: %d' % (num_scenes_train[k]))
    print('num of images: %d' % (num_images_train[k]))
    for i in range(num_classes):
        if count_train[i, k] > 0:
            print('%s: %d' % (classes[i], count_train[i, k]))
    print('==============================')

print('=============testing=================')
print('num of scenes: %d' % (num_scenes_test))
print('num of images: %d' % (num_images_test))
for i in range(num_classes):
    if count_test[i] > 0:
        print('%s: %d' % (classes[i], count_test[i]))
print('==============================')

# write index files
outdir = 'ycb_self_supervision'
filename = osp.join(outdir, 'test.txt')
scenes_test.sort()
with open(filename, 'w') as f:
    for i in range(len(scenes_test)):
        f.write(scenes_test[i])
f.close()

for i in range(num):
    scenes = scenes_train[i]
    filename = osp.join(outdir, 'train_%d.txt' % (i+1))
    scenes.sort()
    with open(filename, 'w') as f:
        for i in range(len(scenes)):
            f.write(scenes[i])
    f.close()


filename = osp.join(outdir, 'all.txt') 
scenes_all.sort()
with open(filename, 'w') as f:
    for i in range(len(scenes_all)):
        f.write(scenes_all[i])
f.close()
