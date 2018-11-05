# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import datasets
import cPickle
import math

class imdb(object):
    """Image database."""

    def __init__(self):
        self._name = ''
        self._num_classes = 0
        self._classes = []
        self._class_colors = []

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path


    def _build_uniform_poses(self):

        self.eulers = []
        for roll in range(0, 360, 15):
            for pitch in range(0, 360, 15):
                for yaw in range(0, 360, 15):
                    self.eulers.append([roll, pitch, yaw])

        # sample indexes
        num_poses = len(self.eulers)
        num_classes = self.num_classes - 1 # no background
        self.pose_indexes = np.zeros((num_classes, ), dtype=np.int32)
        self.pose_lists = []
        for i in range(num_classes):
            self.pose_lists.append(np.random.permutation(np.arange(num_poses)))


    def _build_background_images(self):

        cache_file = os.path.join(self.cache_path, 'backgrounds.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self._backgrounds = cPickle.load(fid)
            print '{} backgrounds loaded from {}'.format(self._name, cache_file)
            return

        backgrounds = []

        '''
        # SUN 2012
        root = os.path.join(self.cache_path, '../SUN2012/data/Images')
        subdirs = os.listdir(root)

        for i in xrange(len(subdirs)):
            subdir = subdirs[i]
            names = os.listdir(os.path.join(root, subdir))

            for j in xrange(len(names)):
                name = names[j]
                if os.path.isdir(os.path.join(root, subdir, name)):
                    files = os.listdir(os.path.join(root, subdir, name))
                    for k in range(len(files)):
                        if os.path.isdir(os.path.join(root, subdir, name, files[k])):
                            filenames = os.listdir(os.path.join(root, subdir, name, files[k]))
                            for l in range(len(filenames)):
                               filename = os.path.join(root, subdir, name, files[k], filenames[l])
                               backgrounds.append(filename)
                        else:
                            filename = os.path.join(root, subdir, name, files[k])
                            backgrounds.append(filename)
                else:
                    filename = os.path.join(root, subdir, name)
                    backgrounds.append(filename)

        # ObjectNet3D
        objectnet3d = os.path.join(self.cache_path, '../ObjectNet3D/data')
        files = os.listdir(objectnet3d)
        for i in range(len(files)):
            filename = os.path.join(objectnet3d, files[i])
            backgrounds.append(filename)

        # AllenCenter
        allencenter = os.path.join(self.cache_path, '../AllenCenter/data')
        subdirs = os.listdir(allencenter)
        for i in xrange(len(subdirs)):
            subdir = subdirs[i]
            files = os.listdir(os.path.join(allencenter, subdir))
            for j in range(len(files)):
                filename = os.path.join(allencenter, subdir, files[j])
                backgrounds.append(filename)
        '''

        # Kinect
        kinect = os.path.join(self.cache_path, '../Kinect')
        files = os.listdir(kinect)
        for i in range(len(files)):
            filename = os.path.join(kinect, files[i])
            backgrounds.append(filename)

        for i in xrange(len(backgrounds)):
            if not os.path.isfile(backgrounds[i]):
                print 'file not exist {}'.format(backgrounds[i])

        self._backgrounds = backgrounds
        print 'build background images finished, {:d} images'.format(len(backgrounds))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(backgrounds, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote backgrounds to {}'.format(cache_file)
