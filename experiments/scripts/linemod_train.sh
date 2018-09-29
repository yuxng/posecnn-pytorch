#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/linemod_flow_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network flownets \
  --pretrained data/checkpoints/flownets_EPE1.951.pth.tar \
  --dataset linemod_train \
  --cfg experiments/cfgs/linemod_flow.yml \
  --cad data/LINEMOD/models.txt \
  --pose data/LINEMOD/poses.txt \
  --solver sgd \
  --epochs 1000
