# DeepSBD for ClipShots
This repository contains our implementation of [deepSBD](https://arxiv.org/abs/1705.08214) for [ClipShots](https://github.com/Tangshitao/ClipShots). The code is modified from [here](https://github.com/kenshohara/3D-ResNets-PyTorch).

## Introduction
We implement deepSBD in this repository. There are 2 backbones that can be selected, including the original Alexnet-like and ResNet-18 introduced in our [paper]().

## Resources
1. The trained model for Alexnet-like backbone. [BaiduYun](https://pan.baidu.com/s/16q3CNuUhLAGkm21PPOqUSg), [Google Drive](https://drive.google.com/open?id=145NCxLhgdrKPIYm-qgp1SRYU_GFmzxxX)
2. The trained model for ResNet-18 backbone. [BaiduYun](https://pan.baidu.com/s/1Bx2uVVQOuEnTxdBBGV3uCQ), [Google Drive](https://drive.google.com/open?id=16nPpsziwhBdCLgSiVuWCIJt-rnXElU6j)
3. The pretrained [model](https://drive.google.com/open?id=10h_axdnkjupEDYe-OiUzm5ALX8w5DX_5) for ResNet-18 backbone.

## Training
Please refer to `train.sh`

## Testing
Add '--no_train' options to `train.sh`.