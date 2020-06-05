#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a object=("vim_mug" "duster" "toy_plane" "black_drill" "blue_mug" "cheezit" "duplo_dude" "graphics_card" "oatmeal_crumble" "orange_drill" "pouch" "remote" "rinse_aid" "tomato_soup" "vegemite")

declare -a set="evaluation"

image_path=""
for j in "${object[@]}"
do
    image_path="$image_path /capri/MOPED_Dataset/data/$j/$set/00/color/000010.jpg"
done

outdir1="output/ycb_object/shapenet_object_train"
outdir2="data/checkpoints"

time ./tools/test_segmentation.py --gpu $1 \
  --image_path "$image_path" \
  --network seg_vgg_embedding \
  --pretrained $outdir1/seg_vgg_embedding_epoch_$2.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_$3.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_vgg_embedding.yml
