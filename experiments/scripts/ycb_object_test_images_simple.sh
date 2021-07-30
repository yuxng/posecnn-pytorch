#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images_simple.py --gpu $1 \
  --imgdir data/images/0320T185355 \
  --network posecnn \
  --depth *-depth.png \
  --color *-color.jpg \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_handle_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_handle.yml
