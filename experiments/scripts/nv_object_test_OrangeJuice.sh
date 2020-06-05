#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/nv_object/nv_object_train/vgg16_nv_object_OrangeJuice_epoch_16.checkpoint.pth \
  --dataset nv_object_test \
  --cfg experiments/cfgs/nv_object_OrangeJuice.yml \
