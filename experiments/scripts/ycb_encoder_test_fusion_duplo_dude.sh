#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py --gpu $1 \
  --network autoencoder \
  --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_fusion_duplo_dude_epoch_150.checkpoint.pth \
  --dataset ycb_encoder_test \
  --cfg experiments/cfgs/ycb_encoder_fusion_duplo_dude.yml \
