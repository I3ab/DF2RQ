# dataset settings
dataset_type = 'C2SegDataset'
data_root = 'data/C2Seg_patch_benchmark/BW'
# train: beijing; val: wuhan

crop_size = (256, 256)
modality_keys = ['hsi', 'msi', 'sar']
train_pipeline = [
    dict(type='LoadMultimodalImageFromFile', 
         modality_keys=modality_keys,
         to_float32=True,
         imdecode_backend='tifffile'),
    dict(type='LoadAnnotations', reduce_zero_label=True, imdecode_backend='tifffile'),
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
         imdecode_backend='tifffile'),
    dict(type='LoadAnnotations', reduce_zero_label=True, imdecode_backend='tifffile'),
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
        data_root=data_root+'/beijing',
        data_prefix=dict(hsi_path='hsi', 
                         msi_path='msi', 
                         sar_path='sar', 
                         seg_map_path='label'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/wuhan',
        data_prefix=dict(hsi_path='hsi', 
                         msi_path='msi',
                         sar_path='sar', 
                         seg_map_path='label'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
