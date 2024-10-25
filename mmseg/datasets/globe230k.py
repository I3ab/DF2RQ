# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
# from mmcv.utils import print_log
import mmengine
import mmengine.fileio as fileio

from prettytable import PrettyTable
from torch.utils.data import Dataset
from PIL import Image

# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

# from mmseg.utils import get_root_logger
from mmseg.registry import DATASETS
from mmseg.datasets.transforms import LoadAnnotations
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Globe230kDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Cropland', 'Forest', 'Grass',
                'Shrub', 'Wetland', 'Water',
                'Tundra', 'Impervious surface', 'Bareland', 
                'Ice/snow'),
        palette=[[252,250,205], [0,123,79], [157,221,106],
                [77,208,159], [111,208,242], [10,78,151], 
                [92,106,55],[155,36,22], [205,205,205], 
                [211,242,255]])
    
    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        # img_dir = self.data_prefix.get('img_path', None)
        rgb_dir = self.data_prefix.get('rgb_path', None)
        dem_dir = self.data_prefix.get('dem_path', None)
        ndvi_dir = self.data_prefix.get('ndvi_path', None)
        vvvh_dir = self.data_prefix.get('vvvh_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(rgb_path=osp.join(rgb_dir, img_name + '.jpg'),
                                 dem_path=osp.join(dem_dir, img_name + self.img_suffix),
                                 ndvi_path=osp.join(ndvi_dir, img_name + self.img_suffix),
                                 vvvh_path=osp.join(vvvh_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=rgb_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(
                    hsi_path=osp.join(rgb_dir, img),
                    msi_path=osp.join(rgb_dir, img.replace('hsi', 'msi')),
                    sar_path=osp.join(rgb_dir, img.replace('hsi', 'sar'))
                )
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len].replace('hsi', 'label') + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['seg_map_path'])
        return data_list
    