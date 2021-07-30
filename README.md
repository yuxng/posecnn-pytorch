# posecnn-pytorch

PyTorch implementation of the PoseCNN and PoseRBPF framework.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

<p align="center"><img src="http://yuxng.github.io/PoseCNN.png" width="640" height="320"/></p>

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find the package is useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        booktitle   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

    @inproceedings{deng2019pose,
        author    = {Xinke Deng and Arsalan Mousavian and Yu Xiang and Fei Xia and Timothy Bretl and Dieter Fox},
        title     = {PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking},
        booktitle = {Robotics: Science and Systems (RSS)},
        year      = {2019}
    }

### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

### Installation

Use python3. If ROS is needed, compile with python2.

1. Install [PyTorch](https://pytorch.org/)

2. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

3. Install Sophus from the Github source code [here](https://github.com/yuxng/Sophus)

4. Install python packages
   ```Shell
   pip install -r requirement.txt
   ```

5. Initialize the submodules in ycb_render
   ```Shell
   git submodule update --init --recursive
   ```

6. Compile the new layers under $ROOT/lib/layers we introduce in PoseCNN
    ```Shell
    cd $ROOT/lib/layers
    sudo python setup.py install
    ```

7. Compile cython components
    ```Shell
    cd $ROOT/lib/utils
    python setup.py build_ext --inplace
    ```

8. Compile the ycb_render in $ROOT/ycb_render
    ```Shell
    cd $ROOT/ycb_render
    sudo python setup.py develop
    ```

### Download

- 3D models of YCB Objects we used [here](https://drive.google.com/file/d/1PTNmhd-eSq0fwSPv0nvQN8h_scR1v-UJ/view?usp=sharing) (3G). Save under $ROOT/data or use a symbol link.

- Our real-world images with pose annotations for 20 YCB objects collected via robot interation [here](https://drive.google.com/file/d/1cQH_dnDzyrI0MWNx8st4lht_q0F6cUrE/view?usp=sharing) (53G). Check our ICRA 2020 [paper](https://arxiv.org/abs/1909.10159) for details.


### Training and testing on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset
    ```Shell
    cd $ROOT/data/YCB_Video
    ln -s $ycb_data data
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # multi-gpu training, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/ycb_video_train.sh

    # testing, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ycb_video_test.sh $GPU_ID

    ```

### Running ROS Kitchen System with YCB Objects
1. Start Kinect for tracking kitchen
    ```Shell
    roslaunch lula_dart multi_kinect.launch
    ```

2. Start DART
    ```Shell
    roslaunch lula_dart kitchen_dart_kinect2.launch
    ```

3. Run DART stitcher
    ```Shell
    ./ros/dart_stitcher_kinect2.py 
    ```

4. Start realsense
    ```Shell
    roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera
    ```

5. Run PoseCNN detection for YCB objects
    ```Shell
    # run posecnn for detection (20 YCB objects and cabinet handle), $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_ycb_object_test_subset_poserbpf_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    ```

6. Run PoseBPRF for YCB objects
    ```Shell
    # $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_poserbpf_ycb_object_test_subset_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    ```

7. (optional) Run PoseCNN detection for blocks
    ```Shell
    # run posecnn for detecting blocks, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_ycb_object_test_subset_poserbpf_realsense.sh $GPU_ID $INSTANCE_ID
    ```

8. (optional) Run PoseBPRF for blocks
    ```Shell
    # $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_poserbpf_ycb_object_test_subset_realsense.sh $GPU_ID $INSTANCE_ID
    ```


