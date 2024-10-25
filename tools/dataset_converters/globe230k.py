# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import shutil
from functools import partial

import numpy as np
from mmengine.utils import (mkdir_or_exist, track_parallel_progress,
                            track_progress)
from PIL import Image
from scipy.io import loadmat
import sys
COCO_LEN = 10000


# def convert_to_trainID(tuple_path, in_img_dir, in_ann_dir, out_img_dir,
#                        out_mask_dir, is_train):
#     imgpath, maskpath = tuple_path
#     shutil.copyfile(
#         osp.join(in_img_dir, imgpath),
#         osp.join(out_img_dir, 'train2014', imgpath) if is_train else osp.join(
#             out_img_dir, 'test2014', imgpath))
#     annotate = loadmat(osp.join(in_ann_dir, maskpath))
#     mask = annotate['S'].astype(np.uint8)
#     mask_copy = mask.copy()
#     for clsID, trID in clsID_to_trID.items():
#         mask_copy[mask == clsID] = trID
#     seg_filename = osp.join(out_mask_dir, 'train2014',
#                             maskpath.split('.')[0] +
#                             '_labelTrainIds.png') if is_train else osp.join(
#                                 out_mask_dir, 'test2014',
#                                 maskpath.split('.')[0] + '_labelTrainIds.png')
#     Image.fromarray(mask_copy).save(seg_filename, 'PNG')

def convert(tuple_path, 
            in_rgb_dir, 
            in_ndvi_dir,
            in_dem_dir,
            in_vvvh_dir,
            in_ann_dir, 
            out_rgb_dir,
            out_ndvi_dir,
            out_dem_dir,
            out_vvvh_dir,
            out_mask_dir, 
            split):
    imgpath, maskpath = tuple_path
    
    shutil.copyfile(osp.join(in_rgb_dir, imgpath.replace('.tif', '.jpg')),
                    osp.join(out_rgb_dir, split, imgpath.replace('.tif', '.jpg')))
    shutil.copyfile(osp.join(in_ndvi_dir, imgpath),
                    osp.join(out_ndvi_dir, split, imgpath))
    shutil.copyfile(osp.join(in_dem_dir, imgpath),
                    osp.join(out_dem_dir, split, imgpath))
    shutil.copyfile(osp.join(in_vvvh_dir, imgpath),
                    osp.join(out_vvvh_dir, split, imgpath))
    shutil.copyfile(osp.join(in_ann_dir, maskpath),
                    osp.join(out_mask_dir, split, maskpath))

def generate_globe230k_list(folder):
    train_list = osp.join(folder, 'train.txt')
    val_list = osp.join(folder, 'val.txt')
    test_list = osp.join(folder, 'test.txt')
    train_paths = []
    val_paths = []
    test_paths = []

    with open(train_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.tif'
            maskpath = basename + '.png'
            train_paths.append((imgpath, maskpath))
            
    with open(val_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.tif'
            maskpath = basename + '.png'
            val_paths.append((imgpath, maskpath))

    with open(test_list) as f:
        for filename in f:
            basename = filename.strip()
            imgpath = basename + '.tif'
            maskpath = basename + '.png'
            test_paths.append((imgpath, maskpath))

    return train_paths, val_list, test_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Stuff 10k annotations to mmsegmentation format')  # noqa
    parser.add_argument('--globe230k_path', default='', help='globe230k path')
    parser.add_argument('--out_dir', default='', help='output path')
    parser.add_argument(
        '--nproc', default=64, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    globe230k_path = args.globe230k_path
    nproc = args.nproc

    out_dir = args.out_dir or globe230k_path
    out_vvvh_dir = osp.join(out_dir, 'vvvh_images')
    out_dem_dir = osp.join(out_dir, 'dem_images')
    out_rgb_dir = osp.join(out_dir, 'rgb_images')
    out_ndvi_dir = osp.join(out_dir, 'ndvi_images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mkdir_or_exist(osp.join(out_vvvh_dir, 'train'))
    mkdir_or_exist(osp.join(out_vvvh_dir, 'val'))
    mkdir_or_exist(osp.join(out_vvvh_dir, 'test'))

    mkdir_or_exist(osp.join(out_dem_dir, 'train'))
    mkdir_or_exist(osp.join(out_dem_dir, 'val'))
    mkdir_or_exist(osp.join(out_dem_dir, 'test'))
    
    mkdir_or_exist(osp.join(out_rgb_dir, 'train'))
    mkdir_or_exist(osp.join(out_rgb_dir, 'val'))
    mkdir_or_exist(osp.join(out_rgb_dir, 'test'))
    
    mkdir_or_exist(osp.join(out_ndvi_dir, 'train'))
    mkdir_or_exist(osp.join(out_ndvi_dir, 'val'))
    mkdir_or_exist(osp.join(out_ndvi_dir, 'test'))
    
    mkdir_or_exist(osp.join(out_mask_dir, 'train'))
    mkdir_or_exist(osp.join(out_mask_dir, 'val'))
    mkdir_or_exist(osp.join(out_mask_dir, 'test'))

    train_list, val_list, test_list = generate_globe230k_list(globe230k_path)

    if args.nproc > 1:
        track_parallel_progress(
            partial(
                convert,
                in_rgb_dir=osp.join(globe230k_path, 'RGB', 'patch_image'),
                in_ndvi_dir=osp.join(globe230k_path, 'NDVI', 'patch_ndvi_selected'),
                in_dem_dir=osp.join(globe230k_path, 'DEM', 'patch_dem_selected'),
                in_vvvh_dir=osp.join(globe230k_path, 'VVVH', 'patch_vvvh_selected'),
                in_ann_dir=osp.join(globe230k_path, 'patch_label'),
                out_rgb_dir=out_rgb_dir,
                out_ndvi_dir=out_ndvi_dir,
                out_dem_dir=out_dem_dir,
                out_vvvh_dir=out_vvvh_dir,
                out_mask_dir=out_mask_dir,
                split='train'),
            train_list,
            nproc=nproc)
        track_parallel_progress(
            partial(
                convert,
                in_rgb_dir=osp.join(globe230k_path, 'RGB', 'patch_image'),
                in_ndvi_dir=osp.join(globe230k_path, 'NDVI', 'patch_ndvi_selected'),
                in_dem_dir=osp.join(globe230k_path, 'DEM', 'patch_dem_selected'),
                in_vvvh_dir=osp.join(globe230k_path, 'VVVH', 'patch_vvvh_selected'),
                in_ann_dir=osp.join(globe230k_path, 'patch_label'),
                out_rgb_dir=out_rgb_dir,
                out_ndvi_dir=out_ndvi_dir,
                out_dem_dir=out_dem_dir,
                out_vvvh_dir=out_vvvh_dir,
                out_mask_dir=out_mask_dir,
                split='val'),
            val_list,
            nproc=nproc)
        track_parallel_progress(
            partial(
                convert,
                in_rgb_dir=osp.join(globe230k_path, 'RGB', 'patch_image'),
                in_ndvi_dir=osp.join(globe230k_path, 'NDVI', 'patch_ndvi_selected'),
                in_dem_dir=osp.join(globe230k_path, 'DEM', 'patch_dem_selected'),
                in_vvvh_dir=osp.join(globe230k_path, 'VVVH', 'patch_vvvh_selected'),
                in_ann_dir=osp.join(globe230k_path, 'patch_label'),
                out_rgb_dir=out_rgb_dir,
                out_ndvi_dir=out_ndvi_dir,
                out_dem_dir=out_dem_dir,
                out_vvvh_dir=out_vvvh_dir,
                out_mask_dir=out_mask_dir,
                split='test'),
            test_list,
            nproc=nproc)
    # else:
    #     track_progress(
    #         partial(
    #             convert_to_trainID,
    #             in_img_dir=osp.join(coco_path, 'images'),
    #             in_ann_dir=osp.join(coco_path, 'annotations'),
    #             out_img_dir=out_img_dir,
    #             out_mask_dir=out_mask_dir,
    #             is_train=True), train_list)
    #     track_progress(
    #         partial(
    #             convert_to_trainID,
    #             in_img_dir=osp.join(coco_path, 'images'),
    #             in_ann_dir=osp.join(coco_path, 'annotations'),
    #             out_img_dir=out_img_dir,
    #             out_mask_dir=out_mask_dir,
    #             is_train=False), test_list)

    print('Done!')


if __name__ == '__main__':
    sys.argv = ['dataset_converters/globe230k.py',]
    main()
