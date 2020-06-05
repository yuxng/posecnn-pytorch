import os
import os.path as osp

classes = ('AlphabetSoup', 'BBQSauce', 'Butter', 'Cherries', 'ChocolatePudding', 
           'Cookies', 'Corn', 'CreamCheese', 'GranolaBars', 'GreenBeans', 'Ketchup', 'MacaroniAndCheese', 
           'Mayo', 'Milk', 'Mushrooms', 'Mustard', 'OrangeJuice', 'Parmesan', 'Peaches', 'PeasAndCarrots', 
           'Pineapple', 'Popcorn', 'Raisins', 'SaladDressing', 'Spaghetti', 'TomatoSauce', 'Tuna', 'Yogurt')

classes_all = ('__background__', 'AlphabetSoup', 'BBQSauce', 'Butter', 'Cherries', 'ChocolatePudding', 
               'Cookies', 'Corn', 'CreamCheese', 'GranolaBars', 'GreenBeans', 'Ketchup', 'MacaroniAndCheese', 
               'Mayo', 'Milk', 'Mushrooms', 'Mustard', 'OrangeJuice', 'Parmesan', 'Peaches', 'PeasAndCarrots', 
               'Pineapple', 'Popcorn', 'Raisins', 'SaladDressing', 'Spaghetti', 'TomatoSauce', 'Tuna', 'Yogurt')

this_dir = osp.dirname(__file__)
root_path = osp.join(this_dir, '..')

for i in range(len(classes)):
    cls = classes[i]

    for ind, c in enumerate(classes_all):
        if c == cls:
            break

    # write training script
    filename = osp.join(root_path, 'experiments', 'scripts', 'nv_object_train_' + cls + '.sh')
    print(filename)
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -x\n')
        f.write('set -e\n')
        f.write('export PYTHONUNBUFFERED="True"\n')
        f.write('export PYTHON_EGG_CACHE=/nfs\n\n')
        f.write('./tools/train_net.py \\\n')
        f.write('  --network posecnn \\\n')
        f.write('  --pretrained data/checkpoints/vgg16-397923af.pth \\\n')
        f.write('  --dataset nv_object_train \\\n')
        f.write('  --cfg experiments/cfgs/nv_object_' + cls + '.yml \\\n')
        f.write('  --solver sgd \\\n')
        f.write('  --epochs 16\n')
    f.close()
    cmd = 'chmod +x ' + filename
    os.system(cmd)

    # write testing script
    filename = osp.join(root_path, 'experiments', 'scripts', 'nv_object_test_' + cls + '.sh')
    print(filename)
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('set -x\n')
        f.write('set -e\n')
        f.write('export PYTHONUNBUFFERED="True"\n')
        f.write('export CUDA_VISIBLE_DEVICES=$1\n\n')
        f.write('./tools/test_net.py --gpu $1 \\\n')
        f.write('  --network posecnn \\\n')
        f.write('  --pretrained output/nv_object/nv_object_train/vgg16_nv_object_' + cls + '_epoch_16.checkpoint.pth \\\n')
        f.write('  --dataset nv_object_test \\\n')
        f.write('  --cfg experiments/cfgs/nv_object_' + cls + '.yml \\\n')
    f.close()
    cmd = 'chmod +x ' + filename
    os.system(cmd)

    # write cfg file
    filename = osp.join(root_path, 'experiments', 'cfgs', 'nv_object_' + cls + '.yml')
    print(filename)
    with open(filename, 'w') as f:
        f.write('EXP_DIR: nv_object\n')
        f.write('INPUT: COLOR\n')
        f.write('TRAIN:\n')
        f.write('  TRAINABLE: True\n')
        f.write('  WEIGHT_DECAY: 0.0001\n')
        f.write('  LEARNING_RATE: 0.001\n')
        f.write('  MILESTONES: !!python/tuple [3]\n')
        f.write('  MOMENTUM: 0.9\n')
        f.write('  BETA: 0.999\n')
        f.write('  GAMMA: 0.1\n')
        f.write('  SCALES_BASE: !!python/tuple [1.0]\n')
        f.write('  IMS_PER_BATCH: 2\n')
        f.write('  NUM_UNITS: 64\n')
        f.write('  HARD_LABEL_THRESHOLD: 0.9\n')
        f.write('  HARD_LABEL_SAMPLING: 0.0\n')
        f.write('  HARD_ANGLE: 5.0\n')
        f.write('  HOUGH_LABEL_THRESHOLD: 100\n')
        f.write('  HOUGH_VOTING_THRESHOLD: 10\n')
        f.write('  HOUGH_SKIP_PIXELS: 10\n')
        f.write('  FG_THRESH: 0.5\n')
        f.write('  FG_THRESH_POSE: 0.5\n')
        f.write('  CLASSES: !!python/tuple [0, ' + str(ind) + ']\n')
        f.write('  SYMMETRY: !!python/tuple [0, 0]\n')
        f.write('  SNAPSHOT_INFIX: nv_object_' + cls + '\n')
        f.write('  SNAPSHOT_EPOCHS: 1\n')
        f.write('  SNAPSHOT_PREFIX: vgg16\n')
        f.write('  USE_FLIPPED: False\n')
        f.write('  CHROMATIC: True\n')
        f.write('  ADD_NOISE: True\n')
        f.write('  VISUALIZE: False\n')
        f.write('  VERTEX_REG: True\n')
        f.write('  POSE_REG: True\n')
        f.write('  # synthetic data\n')
        f.write('  SYNTHESIZE: True\n')
        f.write('  SYNNUM: 40000\n')
        f.write('  SYN_RATIO: 5\n')
        f.write('  SYN_BACKGROUND_SPECIFIC: True\n')
        f.write('  SYN_BACKGROUND_SUBTRACT_MEAN: True\n')
        f.write('  SYN_SAMPLE_OBJECT: False\n')
        f.write('  SYN_SAMPLE_POSE: False\n')
        f.write('  SYN_MIN_OBJECT: 3\n')
        f.write('  SYN_MAX_OBJECT: 5\n')
        f.write('  SYN_TNEAR: 0.4\n')
        f.write('  SYN_TFAR: 1.2\n')
        f.write('  SYN_BOUND: 0.2\n')
        f.write('  SYN_STD_ROTATION: 15\n')
        f.write('  SYN_STD_TRANSLATION: 0.05\n')
        f.write('TEST:\n')
        f.write('  SINGLE_FRAME: True\n')
        f.write('  CLASSES: !!python/tuple [0, ' + str(ind) + ']\n')
        f.write('  SYMMETRY: !!python/tuple [0, 0]\n')
        f.write('  HOUGH_LABEL_THRESHOLD: 400\n')
        f.write('  HOUGH_VOTING_THRESHOLD: 10\n')
        f.write('  HOUGH_SKIP_PIXELS: 10\n')
        f.write('  DET_THRESHOLD: 0.3\n')
        f.write('  SCALES_BASE: !!python/tuple [1.0]\n')
        f.write('  VISUALIZE: False\n')
        f.write('  SYNTHESIZE: True\n')
        f.write('  POSE_REFINE: True\n')
        f.write('  ROS_CAMERA: camera\n')
    f.close()

    # write docker
    filename = osp.join(root_path, 'docker', 'job_nv_object_' + cls + '.json')
    print(filename)
    with open(filename, 'w') as f:
        f.write('{\n')
        f.write('  "jobDefinition": {\n')
        f.write('    "name": "nv object ' + cls + '",\n')
        f.write('    "clusterId": 425,\n')
        f.write('    "dockerImage": "nvidian_general/posecnn-pytorch:0.2",\n')
        f.write('    "jobType": "BATCH",\n')
        f.write('    "command": "cd /nfs/Projects/posecnn-pytorch; ls; sh ./experiments/scripts/nv_object_train_' + cls + '.sh",\n')
        f.write('    "resources": {\n')
        f.write('      "cpuCores": 4,\n')
        f.write('      "gpus": 1,\n')
        f.write('      "systemMemory": 32\n')
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
