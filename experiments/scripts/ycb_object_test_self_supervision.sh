#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_test \
  --cfg experiments/cfgs/ycb_object_self_supervision.yml
