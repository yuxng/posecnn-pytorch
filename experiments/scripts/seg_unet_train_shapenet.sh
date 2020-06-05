#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_unet \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_unet.yml \
  --solver adam \
  --epochs 16
