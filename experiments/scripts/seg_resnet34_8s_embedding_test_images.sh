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
  --imgdir /capri/HOME_DATASET/test/scene_01 \
  --color *.jpg \
  --network seg_resnet34_8s_embedding \
  --network_cor seg_resnet34_8s_triplet \
  --pretrained $outdir1/seg_resnet34_8s_embedding_multi_epoch_16.checkpoint.pth  \
  --pretrained_rrn $outdir1/seg_rrn_unet_epoch_15.checkpoint.pth  \
  --pretrained_cor output/ycb_object/shapenet_rendering_train/seg_resnet34_8s_triplet_cosine_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --codebook data/codebooks/codebook_ycb_embeddings.mat \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml \
