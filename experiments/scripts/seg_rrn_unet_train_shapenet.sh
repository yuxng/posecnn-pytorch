#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network rrn_unet \
  --pretrained data/checkpoints/RRN_OID_checkpoint.pth \
  --dataset shapenet_object_train \
  --cfg experiments/cfgs/seg_rrn_unet.yml \
  --solver adam \
  --epochs 16
