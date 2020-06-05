#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net_mapping.py \
  --network seg_resnet34_8s_mapping \
  --network_ref seg_resnet34_8s_mapping \
  --pretrained_ref output/ycb_object/shapenet_rendering_train/seg_resnet34_8s_triplet_euclidean_epoch_2.checkpoint.pth  \
  --dataset shapenet_encoder_train \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_mapping_euclidean.yml \
  --solver adam \
  --epochs 16
