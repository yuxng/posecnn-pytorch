#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_video_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt \
  --solver sgd \
  --epochs 8
