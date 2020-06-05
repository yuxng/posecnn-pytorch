import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import os.path as osp
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import datasets
from fcn.config import cfg
from torchvision.transforms import transforms
from utils.blob import chromatic_transform, add_noise


class ShapeNetRendering(data.Dataset, datasets.imdb):
    def __init__(self, image_set, shapenet_rendering_path = None):

        self._name = 'shapenet_rendering_' + image_set
        self._image_set = image_set
        self._data_path = self._get_default_path() if shapenet_rendering_path is None \
                            else shapenet_rendering_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self.object_paths, self.object_nums, self.object_names = self.list_objects()
        self._num_classes = len(self.object_paths)
        self.object_lists = np.random.permutation(np.arange(len(self.object_paths)))
        self.object_index = 0
        self._size = cfg.TRAIN.SYNNUM

        self.lb_shift = -0.1
        self.ub_shift = 0.1
        self.lb_scale = 0.5
        self.ub_scale = 1.0

        assert os.path.exists(self._data_path), \
                'shapenet_rendering path does not exist: {}'.format(self._data_path)


    def list_objects(self, root_path=None):
        if root_path is None:
            root_path = self._data_path
        object_paths = []
        object_nums = []
        object_names = []
        # synsets
        synsets = os.listdir(root_path)
        for i in range(len(synsets)):
            synset = synsets[i]
            # objects
            objects = os.listdir(os.path.join(root_path, synset))
            for j in range(len(objects)):
                path = os.path.join(root_path, synset, objects[j])
                files = os.listdir(path)
                object_paths.append(path)
                object_nums.append(len(files))
                object_names.append(objects[j])
        print('%d objects in %s' % (len(object_paths), root_path))
        return object_paths, object_nums, object_names


    def random_crop(self, im, width, height):
        x = random.randint(0, im.shape[1] - width)
        y = random.randint(0, im.shape[0] - height)
        im = im[y:y+height, x:x+width]
        return im


    def transform_image(self, im):
        '''
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)
        im = self.random_crop(im, cfg.TRAIN.SYN_CROP_SIZE, cfg.TRAIN.SYN_CROP_SIZE)
        '''
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor = torch.clamp(im_tensor, min=0.0, max=1.0)
        im_tensor = im_tensor.permute(2, 0, 1)
        return im_tensor


    def _render_item(self):

        # select an object
        if self.object_index >= len(self.object_lists):
            self.object_index = 0
            self.object_lists = np.random.permutation(np.arange(len(self.object_paths)))
        obj_index = self.object_lists[self.object_index]
        object_path = self.object_paths[obj_index]
        image_num = self.object_nums[obj_index]
        self.object_index += 1

        # anchor
        anchor_index = np.random.randint(0, image_num)

        # positive
        while True:
            positive_index = np.random.randint(0, image_num)
            if positive_index != anchor_index:
                break

        # negative
        while True:
            negative_obj_index = np.random.randint(0, len(self.object_paths))
            if negative_obj_index != obj_index:
                break
        negative_object_path = self.object_paths[negative_obj_index]
        negative_image_num = self.object_nums[negative_obj_index]
        negative_index = np.random.randint(0, negative_image_num)

        # read images in BGR order
        filename = os.path.join(object_path, '%06d.jpg' % (anchor_index))
        im = cv2.imread(filename)
        image_anchor_blob = self.transform_image(im)

        filename = os.path.join(object_path, '%06d.jpg' % (positive_index))
        im = cv2.imread(filename)
        image_positive_blob = self.transform_image(im)

        filename = os.path.join(negative_object_path, '%06d.jpg' % (negative_index))
        im = cv2.imread(filename)
        image_negative_blob = self.transform_image(im)

        # class label
        label_positive = torch.zeros((self._num_classes, ), dtype=torch.float32)
        label_positive[obj_index] = 1
        label_negative = torch.zeros((self._num_classes, ), dtype=torch.float32)
        label_negative[negative_obj_index] = 1

        sample = {'image_anchor': image_anchor_blob,
                  'image_positive': image_positive_blob,
                  'image_negative': image_negative_blob,
                  'label_positive': label_positive,
                  'label_negative': label_negative}
        return sample


    def __getitem__(self, index):
        return self._render_item()


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where shapenet_rendering is expected to be installed.
        """
        if self._image_set == 'train':
            return os.path.join(datasets.ROOT_DIR, 'data', 'shapenet_rendering')
        else:
            return os.path.join(datasets.ROOT_DIR, 'data', 'ycb_rendering')
