#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

declare -a arr=("block_green" "block_red" "block_yellow")

for i in "${arr[@]}"
do
    echo "$i"

    # test
    ./tools/test_net.py --gpu 1 \
         --network autoencoder \
         --pretrained data/checkpoints/encoder_ycb_object_self_supervision_train_block_"$i"_epoch_60.checkpoint.pth \
         --dataset ycb_encoder_test \
         --cfg experiments/cfgs/ycb_encoder_"$i".yml

    # copy codebook
    # cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".mat data/codebooks
    # cp output/ycb_object/ycb_encoder_test/ycb_object_"$i"/codebook_ycb_encoder_test_"$i".pth data/codebooks
done
