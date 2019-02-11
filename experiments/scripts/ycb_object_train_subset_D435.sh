#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

LOG="experiments/logs/ycb_object_train_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_pose_epoch_11.checkpoint.pth \
  --startepoch 11 \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset_D435.yml \
  --solver sgd \
  --epochs 16






