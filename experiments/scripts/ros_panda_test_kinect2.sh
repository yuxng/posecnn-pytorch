#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ros_panda_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./ros/test_images.py --gpu $1 \
  --instance $2 \
  --network posecnn \
  --pretrained output/panda/panda_train/vgg16_panda_epoch_5_no_camera.checkpoint.pth \
  --dataset panda_test \
  --cfg experiments/cfgs/panda.yml
