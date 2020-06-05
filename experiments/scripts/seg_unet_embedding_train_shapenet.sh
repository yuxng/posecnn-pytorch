#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_unet_embedding \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_unet_embedding.yml \
  --solver adam \
  --epochs 16
