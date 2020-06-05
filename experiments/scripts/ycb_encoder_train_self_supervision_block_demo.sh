#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_encoders_self_supervision.py \
  --network autoencoder \
  --pretrained data/checkpoints/encoder_ycb_object_self_supervision_train_block_median_cls_epoch_60.checkpoint.pth \
  --dataset ycb_encoder_self_supervision_train_block_median_demo \
  --cfg experiments/cfgs/ycb_encoder_self_supervision_block_demo.yml \
  --solver adam \
  --epochs 60
