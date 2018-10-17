#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,3

LOG="experiments/logs/ycb_video_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_video/ycb_video_train/vgg16_ycb_video_can_box_banana_epoch_5.checkpoint.pth \
  --startepoch 5 \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video_subset.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt \
  --solver sgd \
  --epochs 16
