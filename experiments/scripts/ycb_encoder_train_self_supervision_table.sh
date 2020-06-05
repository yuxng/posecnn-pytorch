#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_encoders_self_supervision.py \
  --network autoencoder \
  --pretrained data/checkpoints/encoder_ycb_object_cls_epoch_200.checkpoint.pth \
  --dataset ycb_encoder_self_supervision_train_table \
  --cfg experiments/cfgs/ycb_encoder_self_supervision_table.yml \
  --solver adam \
  --epochs 60
