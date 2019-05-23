#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

LOG="experiments/logs/ycb_encoder_train_subset.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network autoencoder \
  --dataset ycb_encoder_train \
  --cfg experiments/cfgs/ycb_encoder_subset.yml \
  --solver adam \
  --epochs 200
