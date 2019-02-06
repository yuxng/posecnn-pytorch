#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/panda_test_images.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_images.py --gpu 0 \
  --imgdir data/Images/panda2 \
  --network posecnn \
  --pretrained output/panda/panda_train/vgg16_panda_epoch_8.checkpoint.pth \
  --dataset panda_train \
  --cfg experiments/cfgs/panda.yml
