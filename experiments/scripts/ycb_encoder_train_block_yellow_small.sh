#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network autoencoder \
  --dataset ycb_encoder_train \
  --cfg experiments/cfgs/ycb_encoder_block_yellow_small.yml \
  --solver adam \
  --epochs 200
