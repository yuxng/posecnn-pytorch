import os
import os.path as osp

classes = ('black_drill', 'duplo_dude', 'graphics_card', 'oatmeal_crumble', 'pouch', \
           'rinse_aid', 'vegemite', 'cheezit', 'duster', 'honey', 'orange_drill', 'remote', 'toy_plane', 'vim_mug')

classes_all = ('__background__', 'black_drill', 'duplo_dude', 'graphics_card', 'oatmeal_crumble', 'pouch', \
               'rinse_aid', 'vegemite', 'cheezit', 'duster', 'honey', 'orange_drill', 'remote', 'toy_plane', 'vim_mug')

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..')

for i in range(len(classes)):
    cls = classes[i]

    for ind, c in enumerate(classes_all):
        if c == cls:
            break

    # write training script
    filename = osp.join(root_path, 'experiments', 'scripts', 'moped_encoder_train_' + cls + '.sh')
    print(filename)
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -x\n')
        f.write('set -e\n')
        f.write('export PYTHONUNBUFFERED="True"\n')
        f.write('export PYTHON_EGG_CACHE=/nfs\n\n')
        f.write('./tools/train_net.py \\\n')
        f.write('  --network autoencoder \\\n')
        f.write('  --dataset moped_encoder_train \\\n')
        f.write('  --cfg experiments/cfgs/moped_encoder_' + cls + '.yml \\\n')
        f.write('  --solver adam \\\n')
        f.write('  --epochs 200\n')
    f.close()
    cmd = 'chmod +x ' + filename
    os.system(cmd)

    # write testing script
    filename = osp.join(root_path, 'experiments', 'scripts', 'moped_encoder_test_' + cls + '.sh')
    print(filename)
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -x\n')
        f.write('set -e\n')
        f.write('export PYTHONUNBUFFERED="True"\n')
        f.write('export CUDA_VISIBLE_DEVICES=$1\n\n')
        f.write('./tools/test_net.py --gpu $1 \\\n')
        f.write('  --network autoencoder \\\n')
        f.write('  --pretrained output/moped_object/moped_encoder_train/encoder_moped_object_' + cls + '_epoch_80.checkpoint.pth \\\n')
        f.write('  --dataset moped_encoder_test \\\n')
        f.write('  --cfg experiments/cfgs/moped_encoder_' + cls + '.yml \\\n')
    f.close()
    cmd = 'chmod +x ' + filename
    os.system(cmd)

    # write cfg file
    filename = osp.join(root_path, 'experiments', 'cfgs', 'moped_encoder_' + cls + '.yml')
    print(filename)
    with open(filename, 'w') as f:
        f.write('EXP_DIR: moped_object\n')
        f.write('INPUT: COLOR\n')
        f.write('TRAIN:\n')
        f.write('  TRAINABLE: True\n')
        f.write('  WEIGHT_DECAY: 0.0001\n')
        f.write('  LEARNING_RATE: 0.0002\n')
        f.write('  MILESTONES: !!python/tuple [100]\n')
        f.write('  MOMENTUM: 0.9\n')
        f.write('  BETA: 0.999\n')
        f.write('  GAMMA: 0.1\n')
        f.write('  SCALES_BASE: !!python/tuple [1.0]\n')
        f.write('  IMS_PER_BATCH: 128\n')
        f.write('  NUM_UNITS: 128\n')
        f.write('  CLASSES: !!python/tuple [' + str(ind) + ']\n')
        f.write('  SNAPSHOT_INFIX: moped_object_' + cls + '\n')
        f.write('  SNAPSHOT_EPOCHS: 10\n')
        f.write('  SNAPSHOT_PREFIX: encoder\n')
        f.write('  USE_FLIPPED: False\n')
        f.write('  CHROMATIC: True\n')
        f.write('  ADD_NOISE: True\n')
        f.write('  VISUALIZE: False\n')
        f.write('  BOOSTRAP_PIXELS: 2000\n')
        f.write('  # synthetic data\n')
        f.write('  UNIFORM_POSE_INTERVAL: 5\n')
        f.write('  SYNTHESIZE: True\n')
        f.write('  SYNNUM: 186624\n')
        f.write('  SYN_BACKGROUND_SPECIFIC: False\n')
        f.write('  SYN_SAMPLE_OBJECT: False\n')
        f.write('  SYN_SAMPLE_POSE: False\n')
        f.write('  SYN_WIDTH: 128\n')
        f.write('  SYN_HEIGHT: 128\n')
        f.write('TEST:\n')
        f.write('  SCALES_BASE: !!python/tuple [1.0]\n')
        f.write('  IMS_PER_BATCH: 512\n')
        f.write('  VISUALIZE: False\n')
        f.write('  SYNTHESIZE: True\n')
        f.write('  BUILD_CODEBOOK: True\n')
    f.close()

    # write docker
    filename = osp.join(root_path, 'docker', 'job_moped_encoder_' + cls + '.json')
    print(filename)
    with open(filename, 'w') as f:
        f.write('{\n')
        f.write('  "jobDefinition": {\n')
        f.write('    "name": "encoder ' + cls + '",\n')
        f.write('    "clusterId": 425,\n')
        f.write('    "dockerImage": "nvidian_general/posecnn-pytorch:0.2",\n')
        f.write('    "jobType": "BATCH",\n')
        f.write('    "command": "cd /nfs/Projects/posecnn-pytorch; ls; sh ./experiments/scripts/moped_encoder_train_' + cls + '.sh",\n')
        f.write('    "resources": {\n')
        f.write('      "cpuCores": 8,\n')
        f.write('      "gpus": 2,\n')
        f.write('      "systemMemory": 64\n')
        f.write('    },\n')
        f.write('    "jobDataLocations": [\n')
        f.write('        {\n')
        f.write('                "mountPoint": "/nfs",\n')
        f.write('                "protocol": "NFSV3",\n')
        f.write('                "sharePath": "/export/robot_perception.cosmos608",\n')
        f.write('                "shareHost": "dcg-zfs-03.nvidia.com"\n')
        f.write('        }\n')
        f.write('    ],\n')
        f.write('    "portMappings": []\n')
        f.write('  }\n')
        f.write('}\n')
    f.close()
