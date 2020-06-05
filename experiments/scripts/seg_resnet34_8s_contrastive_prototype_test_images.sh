#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a object=("toy_plane" "pouch" "tomato_soup" "vim_mug" "black_drill" "blue_mug" "cheezit" "duplo_dude" "duster" "graphics_card" "oatmeal_crumble" "orange_drill" "remote" "rinse_aid" "vegemite" "vegemite2")

declare -a set="evaluation"

image_a_path="/capri/HOME_DATASET/fork_00/000190-color.png /capri/HOME_DATASET/fork_00/000190-color.png /capri/HOME_DATASET/fork_00/000190-color.png"
image_b_path="/capri/HOME_DATASET/fork_01/000000-color.png /capri/HOME_DATASET/fork_01/000150-color.png /capri/HOME_DATASET/fork_01/000210-color.png"
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
  --network_cor seg_resnet34_8s_prototype \
  --pretrained $outdir1/seg_resnet34_8s_embedding_epoch_5.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_15.checkpoint.pth  \
  --pretrained_cor $outdir1/seg_resnet34_8s_contrastive_prototype_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml \
  --cfg1 experiments/cfgs/seg_resnet34_8s_contrastive_prototype.yml

