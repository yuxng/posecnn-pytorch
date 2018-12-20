#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export PANGOLIN_WINDOW_URI="headless://"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_object_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_pose_dgx_1_epoch_5.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt
