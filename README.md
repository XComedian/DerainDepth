# DerainDepth
This is the public code of <a href="https://www.mdpi.com/2073-431X/14/1/11">Leveraging Scene Geometry and Depth Information for Robust Image Deraining</a>
## Introduction
In this work, we introduce a novel learning framework that integrates multiple networks: an AutoEncoder for deraining, an auxiliary network to incorporate depth information, and two supervision networks to enforce feature consistency between rainy and clear scenes. This multi-network design enables our model to effectively capture the underlying scene structure, producing clearer and more accurately derained images, leading to improved object detection for autonomous vehicles. Extensive experiments on three widely used datasets demonstrated the effectiveness of our proposed method.

## Prerequisites
Python 3.11.9, torch 2.4.0
Requirements: tensorboard, opencv-python
Platforms: Ubuntu 20.04, cuda-11.2, cudnn 9.1.0.70
## Datasets
We evaluate our model on RainCityScapes (<a href="https://www.cityscapes-dataset.com/downloads/">leftImg8bit_trainval_rain.zip</a>), RainKITTI2012, RainKITTI2015 (<a href="https://pan.baidu.com/s/1sB45qSkCu5q-6Be3ZKLYLA?pwd=1zde">Baidu disk</a>).

## Citation 
```
@article{xu2025leveraging,
  title={Leveraging Scene Geometry and Depth Information for Robust Image Deraining},
  author={Xu, Ningning and Yang, Jidong J},
  journal={Computers},
  volume={14},
  number={1},
  pages={11},
  year={2025},
  publisher={MDPI}
}
```
