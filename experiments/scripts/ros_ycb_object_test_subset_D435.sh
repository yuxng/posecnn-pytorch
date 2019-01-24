#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ros_ycb_object_test_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_images.py --gpu 0 \
  --instance 0 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_pose_dgx_2_epoch_16.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_D435.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt