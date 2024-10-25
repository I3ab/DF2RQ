_base_ = [
    '../_base_/datasets/c2seg/c2seg_bw-pca.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

img_norm_cfg = dict(
    mean=dict(
        hsi_mean=[0.93266386, 0.07038579, 0.0347345, 0.09007076, 0.0111861, 
                  0.02147953, 0.0055078, 0.0153765, -0.01008822, 0.00392935],
        msi_mean=[902.3201592 , 1090.78438656,  989.37881814, 1686.55931541],
        sar_mean=[-19.619953, -27.59927],
    ),
    std=dict(
        hsi_std=[0.34771657, 0.13779184, 0.08190499, 0.03875676, 0.03289428, 
                 0.01630692, 0.01483948, 0.01018066, 0.00604177, 0.00621799],
        msi_std=[285.25549932, 321.02564526, 362.40668622, 609.61312552],
        sar_std=[6.2654386, 6.0273147],
    )
)
crop_size = (256, 256)
data_preprocessor = dict(
    type='MultimodalSegDataPreProcessor',
    size=crop_size,
    norm_cfg=img_norm_cfg,
    pad_val=0,
    seg_pad_val=255
)
norm_cfg = dict(type='SyncBN', requires_grad=True)
hsi_stem = dict(
    bands=10,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])
msi_stem = dict(
    bands=4,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])
sar_stem = dict(
    bands=2,
    hidden_channels=[32, 32],
    kernel_size=[1, 1])

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MultibranchBackbone',
        stem_layers=dict(hsi=hsi_stem, msi=msi_stem, sar=sar_stem),
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
            interpolate_mode='bicubic',
            init_cfg=dict(type='Pretrained',
                          checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth')),
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
optimizer = dict(_delete_=True, 
                 type='AdamW', 
                 lr=3e-5, 
                 betas=(0.9, 0.999),
                 weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=1e-6,
         by_epoch=False,
         begin=0,
         end=500),
    dict(type='CosineAnnealingLR',
         T_max=80000,
         eta_min=1e-7,
         by_epoch=False,
         begin=500,
         end=80000),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

auto_scale_lr = dict(enable=True, base_batch_size=16)
