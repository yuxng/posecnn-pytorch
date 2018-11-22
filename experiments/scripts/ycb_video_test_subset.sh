#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PANGOLIN_WINDOW_URI="headless://"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_video_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_allocentric_pose_epoch_16.checkpoint.pth \
  --dataset ycb_video_val \
  --cfg experiments/cfgs/ycb_video_subset.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt
