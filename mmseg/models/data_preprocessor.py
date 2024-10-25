# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence
import copy
import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)

@MODELS.register_module()
class MultimodalSegDataPreProcessor(BaseDataPreprocessor):
    def __init__(
        self,
        norm_cfg: dict = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        if norm_cfg is not None:
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.mean = norm_cfg['mean']
            self.std = norm_cfg['std']
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def _norm(self, imgs: List[torch.Tensor], mean: List, std: List) -> torch.Tensor:
        normed_imgs = []
        mean = self.cast_data(torch.tensor(mean, dtype=torch.float64))
        std = self.cast_data(torch.tensor(std, dtype=torch.float64))
        stdinv = 1 / std
        for img in imgs:
            c = img.shape[0]
            for i in range(c):
                img[i, :, :] = (img[i, :, :] - mean[i]) * stdinv[i]
            normed_imgs.append(img)
        del imgs, mean, std
        return normed_imgs

    def normalize(self, inputs: dict[List[torch.Tensor]]) -> dict[List[torch.Tensor]]:
        img_keys = [key for key in inputs.keys() if key.endswith("img")]
        for img_m in img_keys:
            mean_m = img_m.replace("img", "mean")
            std_m = img_m.replace("img", "std")
            inputs[img_m] = self._norm(inputs[img_m], self.mean[mean_m], self.std[std_m])
        return inputs

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        # data_samples is shared among all modalities
        data_samples = data.get('data_samples', None)
        img_keys = [key for key in inputs.keys() if key.endswith("img")]
        
        if self._enable_normalize:
            inputs = self.normalize(inputs)

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            # stack batch for each modality
            for i in range(len(img_keys)):
                img_m = img_keys[i]
                # avoid the data_samples being modified multiple times
                if i == 0:
                    inputs[img_m], data_samples = stack_batch(
                        inputs=inputs[img_m],
                        data_samples=data_samples,
                        size=self.size,
                        size_divisor=self.size_divisor,
                        pad_val=self.pad_val,
                        seg_pad_val=self.seg_pad_val)
                else:
                    inputs[img_m], _ = stack_batch(
                        inputs=inputs[img_m],
                        data_samples=None,
                        size=self.size,
                        size_divisor=self.size_divisor,
                        pad_val=self.pad_val,
                        seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            # assert len(set([inputs[key.split('_')[0] + '_img'].shape[:2] for key in img_keys])) == 1 ,\
            #     "the spatial resolution of different modalities should be same"
            # pad images when testing
            if self.test_cfg:
                for img_m in img_keys:
                    inputs[img_m], padded_samples = stack_batch(
                        inputs=inputs[img_m],
                        size=self.test_cfg.get('size', None),
                        size_divisor=self.test_cfg.get('size_divisor', None),
                        pad_val=self.pad_val,
                        seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                for img_m in img_keys:
                    inputs[img_m] = torch.stack(inputs[img_m], dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
