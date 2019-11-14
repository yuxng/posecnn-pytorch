#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

declare -a arr=("black_drill" "duplo_dude" "graphics_card" "oatmeal_crumble" "pouch" "rinse_aid" "vegemite")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 0 \
         --network autoencoder \
         --pretrained output/moped_object/moped_encoder_train/encoder_moped_object_"$i"_epoch_80.checkpoint.pth \
         --dataset moped_encoder_test \
         --cfg experiments/cfgs/moped_encoder_"$i".yml

    # copy codebook
    cp output/moped_object/moped_encoder_test/encoder_moped_object_"$i"_epoch_80_color_particle_50_filter_1/codebook_moped_encoder_test_"$i".mat data/codebooks
    cp output/moped_object/moped_encoder_test/encoder_moped_object_"$i"_epoch_80_color_particle_50_filter_1/codebook_moped_encoder_test_"$i".pth data/codebooks
done
