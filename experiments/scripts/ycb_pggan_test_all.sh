#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py --gpu $1 \
  --network pggan \
  --pretrained output/ycb_object/ycb_encoder_train/pggan_ycb_object_all_epoch_1.checkpoint.pth \
  --dataset ycb_encoder_test \
  --cfg experiments/cfgs/ycb_pggan_all.yml \
