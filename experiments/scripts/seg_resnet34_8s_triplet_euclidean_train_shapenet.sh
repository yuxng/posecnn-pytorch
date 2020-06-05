#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_triplet \
  --dataset shapenet_rendering_train \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_triplet_euclidean.yml \
  --solver adam \
  --epochs 16
