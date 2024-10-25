_base_ = [
    '../_base_/datasets/globe230k/globe230k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
# # use for debug
# vis_backends=[dict(type='LocalVisBackend')]
# visualizer = dict(
#     _delete_=True,
#     type='SegLocalVisualizer', 
#     vis_backends=vis_backends, 
#     name='visualizer')
# # use for debug

# rgb:3,ndvi:1,vvvh:2,dem:1
img_norm_cfg = dict(
    mean=dict(
        rgb_mean=[89.9754, 95.7683, 85.2690],
        dem_mean=[749.5582],
        ndvi_mean=[0.4547],
        vvvh_mean=[-18.0581, -11.1280],
    ),
    std=dict(
        rgb_std=[34.7940,30.9646, 28.6369],
        dem_std=[18.2988],
        ndvi_std=[0.1443],
        vvvh_std=[2.5397, 2.5465],
    )
)
crop_size = (512, 512)
data_preprocessor = dict(
    type='MultimodalSegDataPreProcessor',
    size=crop_size,
    norm_cfg=img_norm_cfg,
    pad_val=0,
    seg_pad_val=255
)
norm_cfg = dict(type='SyncBN', requires_grad=True)
rgb_stem = dict(
    bands=3,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])
# dem_stem = dict(
#     bands=1,
#     hidden_channels=[32, 32],
#     kernel_size=[1, 1])
ndvi_stem = dict(
    bands=1,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])
vvvh_stem = dict(
    bands=2,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MultibranchBackbone',
        stem_layers=dict(rgb=rgb_stem, 
                        #  dem=dem_stem, 
                         ndvi=ndvi_stem, 
                         vvvh=vvvh_stem),
        net_homo=True, # 各模态net是否同构
        share_net=True, # 各模态net是否权重共享
        net=dict(
            type='VisionTransformer',
            img_size=crop_size,
            patch_size=8,
            in_channels=32,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic'),
        norm_cfg=norm_cfg,
        norm_eval=False,
        init_cfg=dict(type='Kaiming', layer='Conv2d')),
    neck=dict(
        type='MultimodalConcat',
        num_modality=3,
        in_channels=[768, 768, 768, 768],
        projector=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
        init_cfg=dict(type='Xavier', layer='Linear')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=13,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=13,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False)
]
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

auto_scale_lr = dict(enable=True, base_batch_size=16)
