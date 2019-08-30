#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_object_test_images.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_images.py --gpu $1 \
  --imgdir data/Images/kitchen3 \
  --network posecnn \
  --pretrained output/ycb_self_supervision/ycb_self_supervision_train/vgg16_ycb_object_self_supervision_epoch_16.checkpoint.pth \
  --dataset ycb_self_supervision_test \
  --cfg experiments/cfgs/ycb_object_self_supervision.yml
