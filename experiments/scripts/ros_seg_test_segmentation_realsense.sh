#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir1="output/ycb_object/shapenet_object_train"

./ros/test_images_segmentation.py --gpu 0 \
  --network seg_resnet34_8s_embedding \
  --pretrained $outdir1/seg_resnet34_8s_embedding_epoch_$2.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_$3.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml
