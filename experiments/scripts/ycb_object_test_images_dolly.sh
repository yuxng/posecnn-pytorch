#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu $1 \
  --imgdir data/images/isaac2 \
  --network posecnn \
  --depth depth/*depth*.png \
  --color color/*color*.png \
  --pretrained data/checkpoints/vgg16_ycb_object_slim_dolly_epoch_16.checkpoint.pth \
  --pretrained_encoder data/checkpoints/encoder_ycb_object_cls_epoch_200.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_dolly2.yml \
  --rand

