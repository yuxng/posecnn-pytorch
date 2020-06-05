#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_object_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --network posecnn \
  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_handle_epoch_3.checkpoint.pth \
  --pretrained_encoder data/checkpoints/encoder_ycb_object_cls_epoch_200.checkpoint.pth \
  --codebook data/codebooks/codebook_ycb_encoder_test_cls \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_handle.yml
