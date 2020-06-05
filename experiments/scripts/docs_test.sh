#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
# export PYTHON_EGG_CACHE=/nfs

./tools/test_net.py \
  --network docsnet \
  --pretrained output/ycb_object/docs_object_train/vgg16_docs_object_epoch_16.checkpoint.pth  \
  --dataset docs_object_test \
  --cfg experiments/cfgs/docs_net.yml
