#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_contrastive \
  --pretrained output/ycb_object/shapenet_object_train/seg_resnet34_8s_contrastive_cosine_epoch_4.checkpoint.pth \
  --startepoch 4 \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_resnet34_8s_contrastive_cosine.yml \
  --solver adam \
  --epochs 16
