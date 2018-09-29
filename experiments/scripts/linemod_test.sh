#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/linemod_flow_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu 0 \
  --network flownets \
  --pretrained output/linemod/linemod_train/flownets_linemod_epoch_200.checkpoint.pth.tar \
  --dataset linemod_train \
  --cfg experiments/cfgs/linemod_flow.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt
