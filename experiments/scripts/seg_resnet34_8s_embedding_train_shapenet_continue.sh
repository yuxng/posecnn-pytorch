#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/ycb_object/shapenet_object_train/seg_resnet34_8s_embedding_multi_epoch_8.checkpoint.pth \
  --startepoch 8 \
  --dataset shapenet_object_train \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml \
  --solver adam \
  --epochs 16
