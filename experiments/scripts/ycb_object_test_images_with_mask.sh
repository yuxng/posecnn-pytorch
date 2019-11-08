#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images_with_mask.py --gpu $1 \
  --imgdir data/Images/duplo_dude \
  --depth depth/*.png \
  --color color/*.jpg \
  --mask mask/*.png \
  --pretrained_encoder output/ycb_object/ycb_encoder_train/encoder_ycb_object_cls_epoch_150.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_fusion.yml
