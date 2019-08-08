#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network autoencoder \
  --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_061_foam_brick_epoch_200.checkpoint.pth \
  --dataset ycb_encoder_self_supervision_train \
  --cfg experiments/cfgs/ycb_encoder_061_foam_brick_self_supervision.yml \
  --solver adam \
  --epochs 100
