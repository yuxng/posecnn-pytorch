#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_images.py --gpu $1 \
  --imgdir data/Images/WeiDataset_20191001_174057 \
  --network posecnn \
  --depth depth_image_*_836212060125.png \
  --color rgb_image_*_836212060125.png \
  --pretrained data/checkpoints/vgg16_ycb_object_blocks_self_supervision_epoch_8.checkpoint.pth \
  --pretrained_encoder data/checkpoints/encoder_ycb_object_self_supervision_train_block_cls_epoch_60.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_blocks_multi_camera.yml
