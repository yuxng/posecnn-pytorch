#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export PYTHON_EGG_CACHE=/nfs

./tools/train_net.py \
  --network docsnet \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset docs_object_train \
  --cfg experiments/cfgs/docs_net.yml \
  --solver adam \
  --epochs 16
