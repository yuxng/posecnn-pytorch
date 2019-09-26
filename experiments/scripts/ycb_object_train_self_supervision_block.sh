#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_blocks_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_train_block \
  --cfg experiments/cfgs/ycb_object_self_supervision_block.yml \
  --solver sgd \
  --epochs 8
