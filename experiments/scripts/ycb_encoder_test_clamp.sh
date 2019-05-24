#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --gpu 0 \
  --network autoencoder \
  --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_clamp_epoch_20.checkpoint.pth \
  --dataset ycb_encoder_test \
  --cfg experiments/cfgs/ycb_encoder_clamp.yml
