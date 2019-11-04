#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

declare -a arr=("block_blue_big" "block_green_big" "block_red_big" "block_yellow_big" \
                "block_blue_median" "block_green_median" "block_red_median" "block_yellow_median" \
                "block_blue_small" "block_green_small" "block_red_small" "block_yellow_small")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 1 \
         --network autoencoder \
         --pretrained data/checkpoints/encoder_ycb_object_"$i"_epoch_200.checkpoint.pth \
         --dataset ycb_encoder_test \
         --cfg experiments/cfgs/ycb_encoder_"$i".yml

    # copy codebook
    cp output/ycb_object/ycb_encoder_test/encoder_ycb_object_"$i"_epoch_200_color_particle_50_filter_1/codebook_ycb_encoder_test_"$i".mat data/codebooks
    cp output/ycb_object/ycb_encoder_test/encoder_ycb_object_"$i"_epoch_200_color_particle_50_filter_1/codebook_ycb_encoder_test_"$i".pth data/codebooks
done
