#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/linemod_flow_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network flownets \
  --pretrained output/linemod/linemod_train/flownets_linemod_epoch_200.checkpoint.pth \
  --dataset linemod_train \
  --cfg experiments/cfgs/linemod_flow_continue.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --solver sgd \
  --epochs 200
