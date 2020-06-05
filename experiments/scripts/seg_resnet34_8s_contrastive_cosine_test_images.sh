#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a object=("toy_plane" "pouch" "tomato_soup" "vim_mug" "black_drill" "blue_mug" "cheezit" "duplo_dude" "duster" "graphics_card" "oatmeal_crumble" "orange_drill" "remote" "rinse_aid" "vegemite" "vegemite2")

declare -a set="evaluation"

declare -a scene_id="000140"

image_a_path="/capri/HOME_DATASET/train/fork_00/000220-color.png"
image_b_path="/capri/HOME_DATASET/test/scene_00/$scene_id-color.png"

image_a_path="$image_a_path /capri/HOME_DATASET/train/jar_00/000130-color.png"
image_b_path="$image_b_path /capri/HOME_DATASET/test/scene_00/$scene_id-color.png"

image_a_path="$image_a_path /capri/HOME_DATASET/train/mug_00/000370-color.png"
image_b_path="$image_b_path /capri/HOME_DATASET/test/scene_00/$scene_id-color.png"

image_a_path="$image_a_path /capri/HOME_DATASET/train/scissors_00/000400-color.png"
image_b_path="$image_b_path /capri/HOME_DATASET/test/scene_00/$scene_id-color.png"

for j in "${object[@]}"
do
    image_a_path="$image_a_path /capri/MOPED_Dataset/data/$j/$set/00/color/000000.jpg"
    image_b_path="$image_b_path /capri/MOPED_Dataset/data/$j/$set/00/color/000035.jpg"
done

outdir1="output/ycb_object/shapenet_object_train"

time ./tools/test_cosegmentation.py --gpu $1 \
  --image_a_path "$image_a_path" \
  --image_b_path "$image_b_path" \
  --network seg_resnet34_8s_embedding \
  --network_cor seg_resnet34_8s_contrastive \
  --pretrained $outdir1/seg_resnet34_8s_embedding_epoch_5.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_15.checkpoint.pth  \
  --pretrained_cor $outdir1/seg_resnet34_8s_contrastive_cosine_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml \
  --cfg1 experiments/cfgs/seg_resnet34_8s_contrastive_cosine.yml

