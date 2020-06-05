#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ros/start_poserbpf.py --gpu $1 \
  --instance $2 \
  --network posecnn \
  --pretrained data/checkpoints/encoder_ycb_object_self_supervision_train_5_cls_epoch_60.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_realsense.yml
