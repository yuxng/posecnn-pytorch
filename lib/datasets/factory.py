# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.ycb_video
import datasets.ycb_object
import datasets.nv_object
import datasets.ycb_self_supervision
import datasets.ycb_encoder
import datasets.moped_encoder
import datasets.linemod
import datasets.isaac_sim
import datasets.background
import numpy as np

# ycb video dataset
for split in ['train', 'val', 'keyframe', 'trainval', 'debug']:
    name = 'ycb_video_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBVideo(split))

# ycb object dataset
for split in ['train', 'test']:
    name = 'ycb_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBObject(split))

# nv object dataset
for split in ['train', 'test']:
    name = 'nv_object_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.NVObject(split))

# ycb self supervision dataset
for split in ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'test', 'all', 'train_block_median', 'train_block_median_azure', 'train_block_median_demo', 'train_block_median_azure_demo', 'train_table',
              'debug', 'train_block', 'train_block_azure', 'train_block_big_sim', 'train_block_median_sim', 'train_block_small_sim']:
    name = 'ycb_self_supervision_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBSelfSupervision(split))

# ycb encoder self supervision dataset
for split in ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'test', 'all', 'train_block_median', 'train_block_median_azure', 'train_block_median_demo', 'train_block_median_azure_demo', 'debug', 'train_table',
              'train_block', 'train_block_azure', 'train_block_big_sim', 'train_block_median_sim', 'train_block_small_sim']:
    name = 'ycb_encoder_self_supervision_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBEncoderSelfSupervision(split))

# ycb encoder dataset
for split in ['train', 'test']:
    name = 'ycb_encoder_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.YCBEncoder(split))

# moped encoder dataset
for split in ['train', 'test']:
    name = 'moped_encoder_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.MOPEDEncoder(split))

# isaac sim dataset
for split in ['train', 'test']:
    name = 'isaac_sim_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.isaac_sim.IsaacSim(split))
            
# linemod dataset
for split in ['train', 'test', 'debug']:
    name = 'linemod_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.linemod(split))

# panda arm dataset
for split in ['train', 'test']:
    name = 'panda_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.panda(split))

# background dataset
for split in ['coco', 'rgbd', 'nvidia', 'table', 'isaac', 'texture']:
    name = 'background_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.BackgroundDataset(split))


def get_dataset(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
