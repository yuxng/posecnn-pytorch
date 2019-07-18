#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_video_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/ycb_video/ycb_video_train/vgg16_ycb_video_dgx_1_epoch_8.checkpoint.pth \
  --dataset ycb_video_keyframe \
  --cfg experiments/cfgs/ycb_video.yml
