import torch
import torchvision
import torch.utils.data as data
import os, math
import os.path as osp
from os.path import *
import numpy as np
import numpy.random as npr
import cv2
import datasets
from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise

class BackgroundDataset(data.Dataset, datasets.imdb):

    def __init__(self, name):

        if name == 'pascal':
            background_dir = os.path.join(self.cache_path, '../PASCAL2012/data')

        # list files
        self.files = []
        if background_dir is not None:
            for filename in os.listdir(background_dir):
                self.files.append(os.path.join(background_dir, filename))

        self.num = len(self.files)
        self.subtract_mean = cfg.TRAIN.SYN_BACKGROUND_SUBTRACT_MEAN
        self._height = cfg.TRAIN.SYN_HEIGHT
        self._width = cfg.TRAIN.SYN_WIDTH
        self._pixel_mean = cfg.PIXEL_MEANS
        print('{} background images'.format(self.num))


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.load(self.files[idx])

    def load(self, filename):

        if np.random.rand(1) < 0.1:
            # constant background image
            background_color = np.ones((self._height, self._width, 3), dtype=np.uint8)
            color = np.random.randint(256, size=3)
            background_color[:, :, 0] = color[0]
            background_color[:, :, 1] = color[1]
            background_color[:, :, 2] = color[2]
        else:
            background_color = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            try:
                # randomly crop a region as background
                bw = background_color.shape[1]
                bh = background_color.shape[0]
                x1 = npr.randint(0, int(bw/3))
                y1 = npr.randint(0, int(bh/3))
                x2 = npr.randint(int(2*bw/3), bw)
                y2 = npr.randint(int(2*bh/3), bh)
                background_color = background_color[y1:y2, x1:x2]
                background_color = cv2.resize(background_color, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
            except:
                background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                print 'bad background image'

            if len(background_color.shape) != 3:
                background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
                print 'bad background_color image'
            if np.random.rand(1) > 0.1:
                background_color = chromatic_transform(background_color)

        if np.random.rand(1) > 0.1:
            background_color = add_noise(background_color)
        background_color = background_color.astype(np.float32)

        if self.subtract_mean:
            background_color -= self._pixel_mean
        background_color = background_color.transpose(2, 0, 1) / 255.0
        return background_color
