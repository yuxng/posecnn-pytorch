#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a object=("black_drill" "blue_mug" "cheezit" "duplo_dude" "duster" "graphics_card" "oatmeal_crumble" "orange_drill" "pouch" "remote" "rinse_aid" "tomato_soup" "toy_plane" "vegemite" "vegemite2" "vim_mug")

declare -a set="evaluation"

image_a_path=""
image_b_path=""
for j in "${object[@]}"
do
    image_a_path="$image_a_path /capri/MOPED_Dataset/data/$j/$set/00/color/000000.jpg"
    image_b_path="$image_b_path /capri/MOPED_Dataset/data/$j/$set/00/color/000000.jpg"
done

time ./tools/test_cosegmentation.py --gpu $1 \
  --image_a_path "$image_a_path" \
  --image_b_path "$image_b_path" \
  --network docsnet \
  --pretrained output/ycb_object/shapenet_object_train/vgg16_docs_object_epoch_16.checkpoint.pth  \
  --dataset shapenet_object_test_coseg \
  --cfg experiments/cfgs/docs_net.yml

