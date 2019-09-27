#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_blocks_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_train \
  --cfg experiments/cfgs/ycb_object_blocks_self_supervision.yml \
  --solver sgd \
  --epochs 8
