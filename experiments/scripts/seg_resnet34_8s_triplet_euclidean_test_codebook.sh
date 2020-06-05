#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_codebook.py --gpu $1 \
  --codebook data/codebooks/codebook_ycb_embeddings_euclidean.mat \
  --cfg experiments/cfgs/seg_resnet34_8s_triplet_euclidean.yml

