#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ros_ycb_object_test_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_images_D435.py --gpu 0 \
  --instance 0 \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_epoch_4.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset.yml
