#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHON_EGG_CACHE=/nfs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset panda_train \
  --cfg experiments/cfgs/panda.yml \
  --solver sgd \
  --epochs 16
