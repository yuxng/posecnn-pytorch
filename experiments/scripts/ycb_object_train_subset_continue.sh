#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=0,3

LOG="experiments/logs/ycb_object_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_pose_dgx_2_epoch_5.checkpoint.pth \
  --startepoch 5 \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset.yml \
  --solver sgd \
  --epochs 16
