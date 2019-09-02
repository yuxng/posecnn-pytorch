#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./ros/test_images_prbpf.py --gpu 0 \
  --instance 0 \
  --network posecnn \
  --pretrained output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_15.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_isaac_sim.yml
