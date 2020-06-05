#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --network seg_vgg \
  --pretrained output/ycb_object/shapenet_object_train/seg_vgg_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_vgg.yml
