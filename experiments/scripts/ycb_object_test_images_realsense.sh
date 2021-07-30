#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

declare -a data="20201022-lmanuelli"
declare -a arr=("840412060917" "836212060125" "839512060362" "841412060263" "932122060857" "932122060861" "932122061900" "932122062010")

for d in data/images/table/"$data"/*; do
    for i in "${arr[@]}"
    do
      time ./tools/test_images.py --gpu 0 \
        --imgdir "$d"/"$i" \
        --meta "$d"/meta.yml \
        --network posecnn \
        --depth aligned_depth_to_color_*.png \
        --color color_*.jpg \
        --pretrained data/checkpoints/vgg16_ycb_object_self_supervision_table_epoch_16.checkpoint.pth \
        --pretrained_encoder data/checkpoints/encoder_ycb_object_self_supervision_table_cls_epoch_60.checkpoint.pth \
        --codebook data/codebooks/codebook_ycb_encoder_test_cls \
        --dataset ycb_object_test \
        --cfg experiments/cfgs/dex_ycb_"$i".yml
    done
done

#  --pretrained data/checkpoints/vgg16_ycb_object_self_supervision_all_epoch_8.checkpoint.pth \
#  --pretrained data/checkpoints/vgg16_ycb_object_self_supervision_train_5_epoch_8.checkpoint.pth \
#  --pretrained output/ycb_object/ycb_object_train/vgg16_ycb_object_slim_epoch_8.checkpoint.pth \
#  --pretrained output/ycb_self_supervision/ycb_self_supervision_train_table/vgg16_ycb_object_self_supervision_table_epoch_16.checkpoint.pth
