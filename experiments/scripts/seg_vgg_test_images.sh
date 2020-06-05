#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a object=("black_drill" "blue_mug" "cheezit" "duplo_dude" "duster" "graphics_card" "oatmeal_crumble" "orange_drill" "pouch" "remote" "rinse_aid" "tomato_soup" "toy_plane" "vegemite" "vegemite2" "vim_mug")

declare -a set="evaluation"

image_path=""
for j in "${object[@]}"
do
    image_path="$image_path /capri/MOPED_Dataset/data/$j/$set/00/color/000000.jpg"
done

time ./tools/test_segmentation.py --gpu $1 \
  --image_path "$image_path" \
  --network seg_vgg \
  --pretrained output/ycb_object/shapenet_object_train/seg_vgg_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_vgg.yml
