#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_video/ycb_video_train/vgg16_ycb_video_dgx_4_epoch_2.checkpoint.pth \
  --startepoch 2 \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video_dgx_4.yml \
  --solver sgd \
  --epochs 8
