#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network autoencoder \
  --dataset moped_encoder_train \
  --cfg experiments/cfgs/moped_encoder_oatmeal_crumble.yml \
  --solver adam \
  --epochs 200
