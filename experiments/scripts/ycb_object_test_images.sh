#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu $1 \
  --imgdir data/Images/000423592912 \
  --network posecnn \
  --depth depth_to_rgb_*.png \
  --color rgb_*.jpg \
  --pretrained data/checkpoints/vgg16_ycb_object_self_supervision_train_5_epoch_8.checkpoint.pth \
  --pretrained_encoder data/checkpoints/encoder_ycb_object_cls_epoch_200.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_self_supervision_test \
  --cfg experiments/cfgs/ycb_object_self_supervision_azure.yml
