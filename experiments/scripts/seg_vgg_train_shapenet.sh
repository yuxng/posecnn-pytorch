#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_vgg \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_vgg.yml \
  --solver adam \
  --epochs 16
