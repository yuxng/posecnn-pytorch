#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_slim_blocks_median_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_train_block_median \
  --cfg experiments/cfgs/ycb_object_self_supervision_block.yml \
  --solver sgd \
  --epochs 8		
