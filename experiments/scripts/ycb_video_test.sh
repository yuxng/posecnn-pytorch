#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_video_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --network flownets \
  --pretrained output/ycb_video/ycb_video_debug/flownets_ycb_video_epoch_1000.checkpoint.pth.tar \
  --dataset ycb_video_debug \
  --cfg experiments/cfgs/ycb_video_flow.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt
