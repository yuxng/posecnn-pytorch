#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network autoencoder \
  --dataset ycb_encoder_self_supervision_train_block_median_sim \
  --cfg experiments/cfgs/ycb_encoder_block_green_median.yml \
  --solver adam \
  --epochs 200
