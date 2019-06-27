#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

LOG="experiments/logs/ycb_object_train_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn_rgbd \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset_D435_more_objs.yml \
  --solver sgd \
  --epochs 48






