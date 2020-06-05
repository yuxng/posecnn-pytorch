#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_prototype \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_resnet34_8s_contrastive_prototype.yml \
  --solver adam \
  --epochs 16
