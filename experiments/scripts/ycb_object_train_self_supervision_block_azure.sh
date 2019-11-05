#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_blocks_median_self_supervision_epoch_8.checkpoint.pth \
  --dataset ycb_self_supervision_train_block_median_azure \
  --cfg experiments/cfgs/ycb_object_self_supervision_block_azure.yml \
  --solver sgd \
  --epochs 8
