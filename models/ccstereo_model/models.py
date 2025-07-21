#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torch.nn as nn
from loguru import logger
from .networks import VisualNet, AudioNet, weights_init
from models.visual.deeplabv3.encoder_decoder import DeepLabV3Plus


class ModelBuilder():
    # builder for visual stream
    def build_visual(self, backbone=18, img_dim=512):
        net = VisualModel(backbone, dim=img_dim)
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, img_dim=512, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc, img_dim=img_dim)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net
    
class VisualModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=2,
        ignore_index=255,
        seg_model="DeepLabV3Plus",
        dim=512,
    ):
        super(VisualModel, self).__init__()
        logger.critical(f"LOADING SEG MODEL <<{seg_model}>>")        
        if backbone == 18:
            backbone_ = torchvision.models.resnet18(True)
        elif backbone == 50:
            backbone_ = torchvision.models.resnet50(True)
        else:
            raise ValueError(f"backbone {backbone} not supported")

        self.backbone = VisualNet(backbone_)
        self.segment = DeepLabV3Plus(
            num_classes=num_classes,
            aspp_in_plane=2048 if backbone == 50 or backbone == 101 else 512,
            aspp_out_plane=dim,
        )

        self.ignore_index = ignore_index

    def forward(self, image):
        x_fea = self.backbone(image)
        out = self.segment(x_fea)
        return out