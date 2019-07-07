#!/bin/bash

DST="/home/yuxiang/GitLab/posecnn-pytorch"

cp $DST/README.md .
cp $DST/lib/datasets/*.py lib/datasets
cp $DST/lib/fcn/*.py lib/fcn 
cp $DST/lib/layers/*.py lib/layers
cp $DST/lib/layers/*.cu lib/layers
cp $DST/lib/layers/*.cpp lib/layers
cp $DST/lib/networks/*.py lib/networks
cp $DST/lib/utils/*.py lib/utils
cp $DST/lib/utils/*.pyx lib/utils
cp $DST/lib/utils/*.c lib/utils
cp $DST/lib/*.py lib
cp $DST/lib/sdf/*.py lib/sdf

cp $DST/experiments/cfgs/*.yml experiments/cfgs
cp $DST/experiments/scripts/*.sh experiments/scripts

cp $DST/ros/*.py ros
cp $DST/tools/*.py tools
cp $DST/ycb_toolbox/*.py ycb_toolbox
cp $DST/ycb_render/*.py ycb_render
cp $DST/ycb_render/*.txt ycb_render
cp $DST/ycb_render/cpp/*.cpp ycb_render/cpp
cp $DST/ycb_render/glutils/*.py ycb_render/glutils
cp $DST/ycb_render/robotPose/*.py ycb_render/robotPose

