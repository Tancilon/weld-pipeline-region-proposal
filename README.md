# GenPose++
[![Website](https://img.shields.io/badge/Website-orange.svg )](https://jiyao06.github.io/Omni6DPose/)
[![Arxiv](https://img.shields.io/badge/Arxiv-green.svg )](https://arxiv.org/pdf/2406.04316.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-v1.0.0-blue.svg)](https://jiyao06.github.io/Omni6DPose/download/)
[![API](https://img.shields.io/badge/API-v0.1.0-blue.svg)](https://github.com/Omni6DPose/Omni6DPoseAPI/)
[![Documentation](https://img.shields.io/badge/Documentation-v0.1.0-blue.svg)](https://jiyao06.github.io/Omni6DPose/cutoop/)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Omni6DPose/GenPose2/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FOmni6DPose%2FGenPose2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

The official implementation of GenPose++, as presented in [Omni6DPose](https://jiyao06.github.io/Omni6DPose/). (ECCV 2024)

![Pipeline](./assets/pipeline.jpg)


## ✨ News
* **2025.08.01**: **A convenient version of GenPose++ with SAM** is released! 🎉
* **2024.08.10**: **GenPose++** is released! 🎉
* **2024.08.01**: **<span style="color: #9AEA27;">Omni6DPose</span>** [dataset](https://github.com/Omni6DPose/Omni6DPoseAPI) and [API](https://github.com/Omni6DPose/Omni6DPoseAPI) are released! 🎉
* **2024.07.01**: **<span style="color: #9AEA27;">Omni6DPose</span>** has been accepted by ECCV2024! 🎉


## 📆 TODO
- [x] Release the Omni6DPose dataset. 
- [x] Release the Omni6DPose API.
- [x] Release the GenPose++ and pretrained models.
- [x] Release a convenient version of GenPose++ with SAM for the downstream tasks.


## ⚙️ Requirements
- Ubuntu 20.04
- Python 3.10.14
- Pytorch 2.1.0
- CUDA 11.8
- 1 * NVIDIA RTX 4090


## 🔨 Installation

- ### Create Environment

```bash
conda create -n genpose2 python==3.10.14
conda activate genpose2
```

- ### Install pytorch

``` bash
conda install cudatoolkit=11
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

- ### Install from requirements.txt

``` bash
pip install -r requirements.txt 
```

- ### Compile pointnet2

``` bash
cd networks/pts_encoder/pointnet2_utils/pointnet2
python setup.py install
```

- ### Compile Cutoop
We provide `cutoop`, a convenient tool for the Omni6DPose dataset. We provide two ways to install it. The detailed installation instructions can be found in the [Omni6DPoseAPI](https://github.com/Omni6DPose/Omni6DPoseAPI/). There we provide the installation instructions using the `pip` package manager.

```bash
sudo apt-get install openexr
pip install cutoop
```

## 🗂️ Download dataset and models

- Download and organize the Omni6DPose dataset by following the instructions provided on the [Omni6DPoseAPI](https://github.com/Omni6DPose/Omni6DPoseAPI) page. Note that the `PAM` dataset and the files `depth_1.zip`, `coord.zip`, and `ir.zip` from the `SOPE` dataset are not required for GenPose++. You may omit downloading these files to save disk space.

- Copy the files from `Meta` to the `$ROOT/configs` directory. The organization of the files should be as follows:

``` bash
genpose2
└──configs
   ├── obj_meta.json
   ├── real_obj_meta.json
   └── config.py
```

- We provide the trained [checkpoints](https://www.dropbox.com/scl/fo/x87lhf7sygjm1gasz153g/AIHBlaGMjhfyW1bKrDe61R4?rlkey=y1f6dqdi40tzcgepccthayudp&st=1sbkxbzf&dl=0). Please download the files to the `$ROOT/results` directory and organize them as follows:

``` bash
genpose2
└──results
   └── ckpts
       ├── ScoreNet
       │   └── scorenet.pth
       ├── EnergyNet
       │   └── energynet.pth
       └── ScaleNet
           └── scalenet.pth
```

## 🚀 Training

Set the parameter `--data_path` in `scripts/train_score.sh`, `scripts/train_energy.sh` and `scripts/train_scale.sh` to your own path of SOPE dataset.

- ### Score network

  Train the score network to generate the pose candidates.

``` bash
bash scripts/train_score.sh
```

- ### Energy network

  Train the energy network to aggragate the pose candidates.

``` bash
bash scripts/train_energy.sh
```

- ### Scale network
  Train the scale network to predict the bounding box length. 
  
  The scale network uses the features extracted by the score network.  You may need to change the parameter `--pretrained_score_model_path` in `scripts/train_scale.sh` if you have trained your own score network.

``` bash
bash scripts/train_scale.sh
```

## 🎯 Inference and evaluation

Set the parameter `--data_path` in `scripts/eval_single.sh` to your own path of ROPE dataset.

- ### Evaluate pose estimation performance

``` bash
bash scripts/eval_single.sh
```

- ### Evaluate pose tracking performance

``` bash
bash scripts/eval_tracking.sh
```

- ### Single video inference and visualization
``` bash
python runners/infer.py
```

## 📷 Real-time camera stream inference
Here we provide a script for real-time camera stream inference with the segmentation masks from [SAM2](https://github.com/Gy920/segment-anything-2-real-time). 

- ### Installation
  First you have to download [SAM2](https://github.com/Gy920/segment-anything-2-real-time) to the base directory, and follow the instruction download the checkpoint `sam2.1_hiera_tiny.pt`.

```bash
pip install -r requirements_camera.txt
```

- ### Inference with RealSense D415 camera
1. Set the `USE_CAM` in the `runners/infer_camera.py` file to `True`. If you want to save the camera stream, set `SAVE_CAM` to `True`.
2. Fill in your camera's serial number in the `CAM_SERIAL_NUM`.
3. If you want to save the inference results, set the `SAVE_RES` to `True`. But note that the inference speed may be affected.
4. The `TRACKING` parameter is used to determine whether use tracking, which means use the pose in last frame as the initial pose. The `TRACKING_T0` parameter is to choose the tracking level. For more details, please see the comments in `runners/infer_camera.py` at the `PARAMETERS` part.
5. Run the script:
``` bash
python runners/infer_camera.py
```

- ### Inference with video stream
1. Download the example data [here](https://www.dropbox.com/scl/fo/o09kj5r1b2bidxsuimh70/AJ9xfeHBMVeLhjUC1HFoqAk?rlkey=wpnyxr17gl1c5enwv0zojqd9f&st=47o4ksfz&dl=0) to `results`, and organize the data structure as follows:
``` bash
results
└── infer_res/0001/video_stream
    ├── *_color.png
    ├── *_depth.exr
    ├── *_mask.exr
    └── *_meta.json
```
2. Set the `USE_CAM` and `SAVE_CAM` in the `runners/infer_camera.py` file to `False`.
3. The other parameters can be the same as [Inference with RealSense D415 camera](#inference-with-realsense-d415-camera).
4. Run the script:
``` bash
python runners/infer_camera.py
```

## 🔖 Citation

If you find our work useful in your research, please consider citing:

``` bash
@article{zhang2024omni6dpose,
  title={Omni6DPose: A Benchmark and Model for Universal 6D Object Pose Estimation and Tracking},
  author={Zhang, Jiyao and Huang, Weiyao and Peng, Bo and Wu, Mingdong and Hu, Fei and Chen, Zijian and Zhao, Bo and Dong, Hao},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}
```

## 📮 Contact

If you have any questions, please feel free to contact us:

[Jiyao Zhang](https://jiyao06.github.io/): [jiyaozhang@stu.pku.edu.cn](mailto:jiyaozhang@stu.pku.edu.cn)

[Weiyao Huang](https://github.com/sshwy): [sshwy@stu.pku.edu.cn](mailto:sshwy@stu.pku.edu.cn)

[Bo Peng](https://github.com/p-b-p-b): [bo.peng@stu.pku.edu.cn](mailto:bo.peng@stu.pku.edu.cn)

[Hao Dong](https://zsdonghao.github.io/): [hao.dong@pku.edu.cn](mailto:hao.dong@pku.edu.cn)

## 📝 License

This project is released under the MIT license. See [LICENSE](LICENSE) for additional details.