#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ros_ycb_object_test_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_images.py --gpu $1 \
  --instance $2 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_pose_dgx_2_epoch_16_5objects.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_kinect2.yml
