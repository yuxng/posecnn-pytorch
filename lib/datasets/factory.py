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
import datasets.ycb_encoder
import datasets.linemod
import datasets.isaac_sim
import datasets.background
import numpy as np

# ycb video dataset
for split in ['train', 'val', 'keyframe', 'trainval', 'debug']:
    name = 'ycb_video_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.YCBVideo(split))

# ycb object dataset
for split in ['train', 'test']:
    name = 'ycb_object_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.YCBObject(split))

# ycb encoder dataset
for split in ['train', 'test']:
    name = 'ycb_encoder_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.YCBEncoder(split))

# isaac sim dataset
for split in ['train', 'test']:
    name = 'isaac_sim_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.isaac_sim.IsaacSim(split))
            
# linemod dataset
for split in ['train', 'test', 'debug']:
    name = 'linemod_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.linemod(split))

# panda arm dataset
for split in ['train', 'test']:
    name = 'panda_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.panda(split))

# background dataset
for split in ['pascal', 'rgbd']:
    name = 'background_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split:
            datasets.BackgroundDataset(split))


def get_dataset(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_datasets():
    """List all registered imdbs."""
    return __sets.keys()
