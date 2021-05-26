# Bilinear3D for Paired Egocentric Interaction Recognition

This repo is the implementation for the paper (https://arxiv.org/abs/2003.10663).

## Overview

This code is based on [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)

## Note
This code was tested with PyTorch 1.0

## dataset
Directories shall be organized as :
`data_root/video_name/img_00001.jpg` rgb

`data_root/video_name/flow_x_00001.jpg` optical flow x

`data_root/video_name/flow_y_00001.jpg` optical flow y

The split file follows the format:
`video_name label`

## Training
run `train_i3d.sh`, please refer to comments in pytorch_i3d.py for arguments.
