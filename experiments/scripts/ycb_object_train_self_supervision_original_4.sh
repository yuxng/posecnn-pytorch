#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON_EGG_CACHE=/nfs

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_original_epoch_4.checkpoint.pth \
  --dataset ycb_self_supervision_train_4 \
  --cfg experiments/cfgs/ycb_object_self_supervision_original_4.yml \
  --solver sgd \
  --epochs 8
