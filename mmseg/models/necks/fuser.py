# Copyright shiyangfeng713@gmail.com, All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.registry import MODELS
from ..utils import resize

@MODELS.register_module()
class MultimodalQueryFusion(BaseModule):
    """
        Multimodal fusion with any number of modalities.
    """
    def __init__(self,
                 num_modality: int,
                 in_channels: List[int],
                 projector: ConfigType,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_modality = num_modality
        self.in_channels = in_channels
        self.fuser = nn.ModuleList()
        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None
        if num_modality > 1:
            for channel in self.in_channels:
                self.fuser.append(
                    nn.Sequential(
                        nn.Linear(self.num_modality*channel, channel),
                        nn.ReLU(inplace=True),
                        nn.Linear(channel, channel),
                        nn.ReLU(inplace=True)))
        else:
            self.fuser = None
            
    def forward(self, inputs: dict[tuple[torch.Tensor]]):
        modalities = list(inputs.keys())
        # For each modality, the number of tuple should be same as the number of in_channels
        assert all([len(inputs[modality])==len(self.in_channels) for modality in modalities]), \
            'The number of tuple should be same as the number of in_channels.'
        outputs = []
        for i in range(len(self.in_channels)):
            # concatenate the features from different modalities in the same feature level l
            feat_l = torch.cat([inputs[modality][i] for modality in modalities], dim=1)            
            # fuse the catenated multimodal features of level l
            if self.num_modality > 1:
                feat_l = self.fuser[i](feat_l.permute(0, 2, 3, 1).contiguous())
                outputs.append(feat_l.permute(0, 3, 1, 2).contiguous())
            else:
                outputs.append(feat_l)
            
        if self.projector is not None:
            outputs = self.projector(tuple(outputs))
            return outputs
        else:
            return tuple(outputs)

@MODELS.register_module()
class MultimodalConcat(BaseModule):
    """
    
    """
    def __init__(self,
                 num_modality: int,
                 in_channels: List[int],
                 projector: ConfigType,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_modality = num_modality
        self.in_channels = in_channels
        self.fuser = nn.ModuleList()
        if projector is not None:
            self.projector = MODELS.build(projector)
        else:
            self.projector = None
        if num_modality > 1:
            for channel in self.in_channels:
                self.fuser.append(
                    nn.Sequential(
                        nn.Linear(self.num_modality*channel, channel),
                        nn.ReLU(inplace=True),
                        nn.Linear(channel, channel),
                        nn.ReLU(inplace=True)))
        else:
            self.fuser = None
            
    def forward(self, inputs: dict[tuple[torch.Tensor]]):
        modalities = list(inputs.keys())
        # For each modality, the number of tuple should be same as the number of in_channels
        assert all([len(inputs[modality])==len(self.in_channels) for modality in modalities]), \
            'The number of tuple should be same as the number of in_channels.'
        outputs = []
        for i in range(len(self.in_channels)):
            # concatenate the features from different modalities in the same feature level l
            feat_l = torch.cat([inputs[modality][i] for modality in modalities], dim=1)            
            # fuse the catenated multimodal features of level l
            if self.num_modality > 1:
                feat_l = self.fuser[i](feat_l.permute(0, 2, 3, 1).contiguous())
                outputs.append(feat_l.permute(0, 3, 1, 2).contiguous())
            else:
                outputs.append(feat_l)
            
        if self.projector is not None:
            outputs = self.projector(tuple(outputs))
            return outputs
        else:
            return tuple(outputs)
