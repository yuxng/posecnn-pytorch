#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network seg_resnet34_8s_triplet \
  --pretrained output/ycb_object/shapenet_rendering_train/seg_resnet34_8s_triplet_cosine_epoch_$2.checkpoint.pth  \
  --dataset shapenet_rendering_test \
  --cfg experiments/cfgs/seg_resnet34_8s_triplet_cosine.yml \

