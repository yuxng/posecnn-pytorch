#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/view.py --gpu $1 \
  --imgdir data/images/isaac2 \
  --depth depth/*depth*.png \
  --color color/*color*.png \
  --cfg experiments/cfgs/ycb_object_subset_dolly2.yml

