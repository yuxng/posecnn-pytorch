#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_encoders_self_supervision.py \
  --network autoencoder \
  --pretrained data/checkpoints/encoder_ycb_object_cls_epoch_200.checkpoint.pth \
  --dataset ycb_encoder_self_supervision_all \
  --cfg experiments/cfgs/ycb_encoder_self_supervision_cabinet_handle.yml \
  --solver adam \
  --epochs 60
