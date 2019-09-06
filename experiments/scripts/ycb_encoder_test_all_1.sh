#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

declare -a arr=("002_master_chef_can" "003_cracker_box" "004_sugar_box" "005_tomato_soup_can" "006_mustard_bottle" \
                "007_tuna_fish_can" "008_pudding_box" "009_gelatin_box" "010_potted_meat_can" "011_banana" "019_pitcher_base" "block_red" "block_green")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 0 \
         --network autoencoder \
         --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_"$i"_epoch_200.checkpoint.pth \
         --dataset ycb_encoder_test \
         --cfg experiments/cfgs/ycb_encoder_"$i".yml

    # copy codebook
    cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".mat data/codebooks
    cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".pth data/codebooks
done
