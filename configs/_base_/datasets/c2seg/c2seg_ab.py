# dataset settings
dataset_type = 'C2SegDataset'
data_root = 'data/C2Seg_patch/AB'
# train: augsburg; val: berlin

crop_size = (128, 128)
train_pipeline = [
    dict(type='LoadMultimodalImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True, imdecode_backend='tifffile'),
    dict(type='PackMultimodalSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True, imdecode_backend='tifffile'),
    dict(type='PackMultimodalSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/augsburg',
        data_prefix=dict(
            hsi_path='hsi', 
            msi_path='msi',
            sar_path='sar',
            seg_map_path='label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root+'/berlin',
        data_prefix=dict(
            hsi_path='hsi', 
            msi_path='msi',
            sar_path='sar',
            seg_map_path='label'),
        pipeline=train_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator



    # dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    # dict(type='DefaultFormatBundle'),
