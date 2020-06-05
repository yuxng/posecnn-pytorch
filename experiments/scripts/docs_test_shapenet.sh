#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/test_net.py \
  --network docsnet \
  --pretrained output/ycb_object/shapenet_object_train/vgg16_docs_object_epoch_7.checkpoint.pth  \
  --dataset shapenet_object_test_coseg \
  --cfg experiments/cfgs/docs_net.yml
