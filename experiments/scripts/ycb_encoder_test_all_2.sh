#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

declare -a arr=("021_bleach_cleanser" "024_bowl" "025_mug" "035_power_drill" "036_wood_block" "037_scissors" "040_large_marker" \
                "051_large_clamp" "052_extra_large_clamp" "061_foam_brick" "block_blue" "block_yellow")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 1 \
         --network autoencoder \
         --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_"$i"_epoch_200.checkpoint.pth \
         --dataset ycb_encoder_test \
         --cfg experiments/cfgs/ycb_encoder_"$i".yml

    # copy codebook
    cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".mat data/codebooks
    cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".pth data/codebooks
done
