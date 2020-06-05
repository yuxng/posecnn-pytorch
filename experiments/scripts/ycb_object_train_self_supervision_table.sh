#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

./tools/train_net_self_supervision.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_slim_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_train_table \
  --cfg experiments/cfgs/ycb_object_self_supervision_table.yml \
  --solver sgd \
  --epochs 16
