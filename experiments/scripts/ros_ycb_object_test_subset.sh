#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ros_ycb_object_test_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_images.py --gpu 0 \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_pose_epoch_5.checkpoint.pth \
  --dataset ycb_object_train \
  --cfg experiments/cfgs/ycb_object_subset.yml \
  --cad data/ycb_object/models.txt \
  --pose data/ycb_object/poses.txt
