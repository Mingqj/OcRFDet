# OcRFDet: Object-Centric Radiance Fields for Multi-View 3D Object Detection in Autonomous Driving

<div align="center">

[Mingqian Ji](https://github.com/Mingqj) </sup>,
[Shanshan Zhang](https://shanshanzhang.github.io/) ✉</sup>,
[Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN) </sup>

Nanjing University of Science and Technology

✉ Corresponding author

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2506.23565)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

## 📖 About

This repository represents the official implementation of the paper titled "OcRFDet: Object-Centric Radiance Fields for Multi-View 3D Object Detection in Autonomous Driving".

We propose Object-Centric Radiance Fields (OcRF) to enhance multi-view 3D object detection by focusing rendering on foreground objects and filtering out background noise. An auxiliary rendering task improves 3D voxel features, while the generated opacity maps are used to refine BEV features through Height-aware Opacity-based Attention (HOA). This explicit geometry-aware design significantly boosts the detector’s ability to understand 3D structure from multi-view RGB images.

## 💾 Main Results

# nuScenes val set
| Config                                                            | mAP  | NDS | Model|
|:-----------------------------------------------------------------:|:----:|:---:|:----:|
| [**OcRFDet**](configs/ocrfdet/ocrfdet.py) | 40.0 | 50.9 |-|

# nuScenes test set
| Config                                                            | mAP  | NDS | Model|
|:-----------------------------------------------------------------:|:----:|:---:|:----:|
| [**OcRFDet**] | 57.2 | 64.8 |-|
