#!/bin/bash
	
set -x
set -e

./ros/data_collection_pose_rbpf.py \
  --instance $1 \
  --target_obj block_blue \
  --world experiments/cfgs/world.yaml
