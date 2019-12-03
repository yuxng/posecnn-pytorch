#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/start_depth_masks.py --gpu $1 \
  --instance $2 \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_blocks_realsense.yml
