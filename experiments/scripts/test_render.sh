#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_render.py --gpu 0 \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt
