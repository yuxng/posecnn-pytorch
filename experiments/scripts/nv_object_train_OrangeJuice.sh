#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network posecnn \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset nv_object_train \
  --cfg experiments/cfgs/nv_object_OrangeJuice.yml \
  --solver sgd \
  --epochs 16
