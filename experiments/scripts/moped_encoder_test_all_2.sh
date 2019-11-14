#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

declare -a arr=("cheezit" "duster" "honey" "orange_drill" "remote" "toy_plane" "vim_mug")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 1 \
         --network autoencoder \
         --pretrained output/moped_object/moped_encoder_train/encoder_moped_object_"$i"_epoch_80.checkpoint.pth \
         --dataset moped_encoder_test \
         --cfg experiments/cfgs/moped_encoder_"$i".yml

    # copy codebook
    cp output/moped_object/moped_encoder_test/encoder_moped_object_"$i"_epoch_80_color_particle_50_filter_1/codebook_moped_encoder_test_"$i".mat data/codebooks
    cp output/moped_object/moped_encoder_test/encoder_moped_object_"$i"_epoch_80_color_particle_50_filter_1/codebook_moped_encoder_test_"$i".pth data/codebooks
done
