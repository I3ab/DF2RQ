# Multimodal Backbone (mmb) network for multimodal remote sensing image
from typing import List, Tuple, Dict
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmseg.registry import MODELS
# from ..utils import ResLayer

# Multibranch Backbone (mbb) for multimodal segmentation
@MODELS.register_module()
class MultibranchBackbone(BaseModule):
    """
    Multibranch Backbone framework for multimodal segmentation.
    The Multibranch Backbone is consist of several modal-specific stem 
    layer and some backbone net. For each modality, the backbone can be
    separated into two parts: stem layer and backbone net.
    
    The stem layer is used to process the input of each modality in 
    different number of channels, and the backbone net is used to extract 
    the features. The backbone net can be initialized with the pre-trained
    weights, and the stem layer can not.
    
    The net is used to extract the features of each modality, and the structure
    of each net can be heterogeneous(异构)/homogeneous(同构).
    If the structure of each net is the same, the share_net can be True or False.
    If the structure of each net is different, the share_net should be False.
    """
    def __init__(self, 
                 stem_layers: dict,
                 share_net: bool,
                 net_homo: bool,
                 net: dict,
                 norm_cfg: dict = dict(type='BN'),
                 norm_eval: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.modalities = list(stem_layers.keys())
        self.num_modality = len(self.modalities)
        assert share_net or self.num_modality==len(net), \
            'if share_net is False, the number of net should be same as num_modality.'
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        
        # bulid stem layer
        self.stem_layers = nn.ModuleDict()
        for modality in self.modalities:
            self.stem_layers[modality] = self._build_stem_layers(**stem_layers[modality])
        
        # build net
        if share_net and net_homo:
            self.net = MODELS.build(net)
        else:
            self.net = nn.ModuleDict()
            for modality in self.modalities:
                self.net[modality] = MODELS.build(net[modality])
                
    def _build_stem_layers(self, 
                           bands: int,
                           hidden_channels: list[int],
                           kernel_size: list[int],):
        """
        
        """
        assert len(hidden_channels) == len(kernel_size), \
            'The length of hidden_channels should be same as kernel_size.'
        stem_layers = nn.Sequential()
        num_layers = len(hidden_channels)
        for i in range(num_layers):
            stem_layers.append(
                ConvModule(
                    bands if i==0 else hidden_channels[i-1],
                    hidden_channels[i],
                    kernel_size=kernel_size[i],
                    norm_cfg=self.norm_cfg,))
        return stem_layers
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): input data of each modality.
        Returns:
            Dict[str, torch.Tensor]: output feature of each modality.
        """
        stem_features = {}
        output = {}
        # stem layer froward
        for modality in self.modalities:
            stem_features[modality] = self.stem_layers[modality](x[modality+'_img'])
        # net forward
        if isinstance(self.net, nn.ModuleDict):
            for modality in self.modalities:
                output[modality] = self.net[modality](stem_features[modality])
        else:
            for modality in self.modalities:
                output[modality] = self.net(stem_features[modality])
        return output
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    
@MODELS.register_module()
class Uni3DPatchBackbone(BaseModule):
    """
    Proposed in SpectralGPT: Spectral Foundation Model
    """
    ...
