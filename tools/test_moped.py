#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a PoseCNN on images"""

import os
import sys

classes = ('black_drill', 'duplo_dude', 'graphics_card', 'oatmeal_crumble', 'pouch', \
           'rinse_aid', 'vegemite', 'cheezit', 'duster', 'orange_drill', 'remote', 'toy_plane', 'vim_mug')

sequences = (5, 7, 6, 5, 5, 6, 5, 6, 5, 5, 6, 5, 5)

if __name__ == '__main__':
    for i in range(len(classes)):
        cls = classes[i]
        print(i, cls, sequences[i])
        for j in range(sequences[i]):
            command = 'python tools/test_images_with_mask.py --gpu 0 --cls_name {} --imgdir data/MOPED/data/{}/evaluation/{:02d} --depth depth/*.png --color color/*.jpg --mask mask/*.png  --pretrained_encoder output/moped_object/moped_encoder_train/encoder_moped_object_{}_epoch_80.checkpoint.pth --codebook data/codebooks/codebook_moped_encoder_test_{} --dataset moped_encoder_test --cfg experiments/cfgs/moped_object.yml'.format(cls, cls, j, cls, cls)
            print(command)
            os.system(command)

            
