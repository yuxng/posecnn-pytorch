#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_object_train \
  --dataset_background background_nvidia \
  --cfg experiments/cfgs/ycb_object_subset_dolly.yml \
  --solver sgd \
  --epochs 16
