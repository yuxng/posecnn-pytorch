#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py --gpu $1 \
  --network autoencoder \
  --pretrained output/moped_object/moped_encoder_train/encoder_moped_object_honey_epoch_80.checkpoint.pth \
  --dataset moped_encoder_test \
  --cfg experiments/cfgs/moped_encoder_honey.yml \
