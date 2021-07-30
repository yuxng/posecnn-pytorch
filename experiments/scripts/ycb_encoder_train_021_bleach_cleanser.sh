#!/bin/bash

set -x
set -e

./tools/train_net.py \
  --network autoencoder \
  --dataset ycb_encoder_train \
  --cfg experiments/cfgs/ycb_encoder_021_bleach_cleanser.yml \
  --solver adam \
  --epochs 200
