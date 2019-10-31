#!/bin/bash

DST="/home/yuxiang/GitLab/posecnn-pytorch"

cp README.md $DST
cp lib/datasets/*.py $DST/lib/datasets
cp lib/fcn/*.py $DST/lib/fcn
cp lib/layers/*.py $DST/lib/layers
cp lib/layers/*.cu $DST/lib/layers
cp lib/layers/*.cpp $DST/lib/layers
cp lib/networks/*.py $DST/lib/networks
cp lib/sdf/*.py $DST/lib/sdf
cp lib/utils/*.py  $DST/lib/utils
cp lib/utils/*.pyx  $DST/lib/utils
cp lib/utils/*.c  $DST/lib/utils
cp lib/*.py $DST/lib

cp experiments/cfgs/*.yml $DST/experiments/cfgs
cp experiments/scripts/*.sh $DST/experiments/scripts
cp msg/* $DST/msg

cp ros/*.py $DST/ros
cp tools/*.py $DST/tools
cp ycb_toolbox/*.py $DST/ycb_toolbox
cp ycb_render/*.py $DST/ycb_render
cp ycb_render/*.txt $DST/ycb_render
cp ycb_render/cpp/*.cpp $DST/ycb_render/cpp
cp ycb_render/glutils/*.py $DST/ycb_render/glutils
cp ycb_render/robotPose/*.py $DST/ycb_render/robotPose
cp ycb_render/robotPose/panda_arm_models/* $DST/ycb_render/robotPose/panda_arm_models
