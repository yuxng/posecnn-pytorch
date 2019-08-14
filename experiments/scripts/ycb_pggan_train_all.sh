#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network pggan \
  --dataset ycb_encoder_train \
  --cfg experiments/cfgs/ycb_pggan_all.yml \
  --solver adam \
  --epochs 200
