# dataset settings
dataset_type = 'Globe230kDataset'
data_root = 'data/Globe230k_v3'
# train: beijing; val: wuhan

crop_size = (512, 512)
modality_keys = ['dem'] # ['rgb', 'ndvi', 'vvvh', 'dem']
train_pipeline = [
    dict(type='LoadMultimodalImageFromFile', 
         modality_keys=modality_keys,
         to_float32=True,
         imdecode_backend='pillow'), # pillow is faster than tifffile
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='MultimodalRandomResize', 
         scale=crop_size,
         ratio_range=(0.5, 2.0), 
         resize_type='MultimodalResize'),
    dict(type='MultimodalRandomCrop', cat_max_ratio=0.75, crop_size=crop_size),
    dict(type='MultimodalRandomFlip', prob=0.5),
    dict(type='PackMultimodalSegInputs')
]
test_pipeline = [
    dict(type='LoadMultimodalImageFromFile', 
         modality_keys=modality_keys,
         to_float32=True,
         imdecode_backend='pillow'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='MultimodalResize', scale=crop_size),
    dict(type='PackMultimodalSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(rgb_path='image_patch', 
                         dem_path='dem_patch', 
                         ndvi_path='ndvi_patch', 
                         vvvh_path='vvvh_patch',
                         seg_map_path='label_patch'),
        ann_file='train_num.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(rgb_path='image_patch', 
                         dem_path='dem_patch', 
                         ndvi_path='ndvi_patch', 
                         vvvh_path='vvvh_patch',
                         seg_map_path='label_patch'),
        ann_file='val_num.txt',
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(rgb_path='image_patch', 
                         dem_path='dem_patch', 
                         ndvi_path='ndvi_patch', 
                         vvvh_path='vvvh_patch',
                         seg_map_path='label_patch'),
        ann_file='test_num.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
