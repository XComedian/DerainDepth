# DerainDepth
This is the public code of <a href="https://www.mdpi.com/2073-431X/14/1/11">Leveraging Scene Geometry and Depth Information for Robust Image Deraining</a>
## Introduction
In this work, we introduce a novel learning framework that integrates multiple networks: an AutoEncoder for deraining, an auxiliary network to incorporate depth information, and two supervision networks to enforce feature consistency between rainy and clear scenes. This multi-network design enables our model to effectively capture the underlying scene structure, producing clearer and more accurately derained images, leading to improved object detection for autonomous vehicles. Extensive experiments on three widely used datasets demonstrated the effectiveness of our proposed method.

## Datasets
We evaluate our model on <a href="https://github.com/xw-hu/DAF-Net">RainCityScapes</a>, <a href="https://pan.baidu.com/s/1sB45qSkCu5q-6Be3ZKLYLA?pwd=1zde">RainKITTI2012, RainKITTI2015</a>

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
