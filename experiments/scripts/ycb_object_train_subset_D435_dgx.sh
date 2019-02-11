#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON_EGG_CACHE=/NFS

time ./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_pose_epoch_11.checkpoint.pth \
  --startepoch 11 \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset_D435.yml \
  --solver sgd \
  --epochs 16






