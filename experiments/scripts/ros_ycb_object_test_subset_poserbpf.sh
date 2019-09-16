#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./ros/test_images_prbpf.py --gpu 0 \
  --instance 0 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_blocks_self_supervision_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_prbpf.yml

#  --pretrained_compare output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_epoch_16.checkpoint.pth \
#  --pretrained_compare output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_4.checkpoint_old.pth \
#  --pretrained_compare output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_16.checkpoint.pth \

#  --pretrained output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_10.checkpoint.pth \

#  --pretrained_compare output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_epoch_16.checkpoint.pth \


#  --pretrained output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_16.checkpoint.pth \
