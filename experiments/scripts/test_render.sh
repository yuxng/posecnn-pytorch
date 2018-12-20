#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./ycb_toolbox/test_render.py --gpu 0 \
  --cad data/YCB_Video/models.txt.selected \
  --pose data/YCB_Video/poses.txt.selected
