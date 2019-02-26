#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_pose_dgx_1_epoch_11.checkpoint.pth \
  --startepoch 11 \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset_dgx_1.yml \
  --solver sgd \
  --epochs 16
