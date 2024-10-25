# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type
import sys
import mmcv
import torch
import torch.nn as nn
import numpy as np
from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import init_model, mm_inference_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer

class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result):
    seg_visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='WandbVisBackend')],
        save_dir='temp_dir',
        alpha=1.0)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=[[0,255,255], [255,255,255], [255,0,0],
                [221,160,221], [148,0,211], [255,0,255], 
                [255, 255, 0],[205,133,63], [189,183,107], 
                [0,255,0], [154,205,50], [139,69,19], [72,61,139]])

    image = np.ones((256, 256, 3), dtype=np.uint8) * 255

    seg_visualizer.add_datasample(
        name='predict',
        image=image,
        data_sample=result,
        draw_gt=False,
        draw_pred=True,
        with_labels=False,
        wait_time=0,
        out_file=None,
        show=False)
    print('result', result.pred_sem_seg.data)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0].squeeze(0)  # remove the batch
        overlaid_img = np.zeros_like(image) + 255
        drawn_img = seg_visualizer.draw_featmap(
            feature, 
            overlaid_img,
            channel_reduction='select_max',
            alpha=1.0)
        seg_visualizer.add_image(f'feature_map{i}', drawn_img)

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        sem_seg = torch.from_numpy(sem_seg) -1
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        data_sample = SegDataSample()
        data_sample.gt_sem_seg = gt_mask
        print('gt_mask', gt_mask)

        seg_visualizer.add_datasample(
            name='gt_mask',
            image=image,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=False,
            wait_time=0,
            with_labels=False,
            out_file=None,
            show=False)

    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('filename', help='Image filename file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    source = [
        'backbone.net',
    ]
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break
    """
    'rgb_path':
    'data/Globe230k_v3/image_patch/data_1.jpg'
    'dem_path':
    'data/Globe230k_v3/dem_patch/data_1.tif'
    'ndvi_path':
    'data/Globe230k_v3/ndvi_patch/data_1.tif'
    'vvvh_path':
    'data/Globe230k_v3/vvvh_patch/data_1.tif'
    'seg_map_path':
    'data/Globe230k_v3/label_patch/data_1.png'
    """
    imgs = dict(
        hsi_path=args.data_dir + '/hsi/' + args.filename + '_hsi' + '.tiff',
        msi_path=args.data_dir + '/msi/' + args.filename + '_msi' + '.tiff',
        sar_path=args.data_dir + '/sar/' + args.filename + '_sar' + '.tiff',
        seg_map_path=args.data_dir + '/label/' + args.filename + '_label' + '.tiff',
    )
    with recorder:
        # test a single image, and record feature map to data_buffer
        result = mm_inference_model(model, imgs)

    visualize(args, model, recorder, result)


if __name__ == '__main__':

    main()
