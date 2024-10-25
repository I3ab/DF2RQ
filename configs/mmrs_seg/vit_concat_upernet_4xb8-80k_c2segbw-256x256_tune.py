_base_ = [
    '../_base_/datasets/c2seg/c2seg_bw.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_80k.py'
]

img_norm_cfg = dict(
    mean=dict(
        hsi_mean=[0.09986721, 0.09861182, 0.09736297, 0.09601279, 0.09466794,
       0.09346812, 0.09232572, 0.09145092, 0.09066506, 0.09041355,
       0.09079242, 0.09058639, 0.09014223, 0.09003624, 0.09013505,
       0.09044274, 0.09100031, 0.09145813, 0.09192843, 0.09248866,
       0.09302586, 0.09374638, 0.09447818, 0.09596613, 0.09581663,
       0.09557939, 0.09527811, 0.09537612, 0.09529296, 0.09469415,
       0.09434015, 0.09428018, 0.09479886, 0.09541113, 0.09587669,
       0.09687101, 0.09738953, 0.09832127, 0.09895454, 0.09837405,
       0.09497084, 0.09182249, 0.08881705, 0.08616859, 0.08242726,
       0.07877567, 0.07613014, 0.07422501, 0.07333961, 0.07317502,
       0.07327873, 0.07411174, 0.07526539, 0.07743033, 0.0793338 ,
       0.08003521, 0.07987303, 0.08032965, 0.08154666, 0.082027  ,
       0.08531117, 0.0868018 , 0.08683132, 0.08657411, 0.08757851,
       0.08701074, 0.08342319, 0.08017913, 0.07667795, 0.07393184,
       0.07131778, 0.07014034, 0.0695982 , 0.07120184, 0.0762165 ,
       0.08261579, 0.08949149, 0.0920078 , 0.09389601, 0.09709487,
       0.10052502, 0.10081016, 0.09890448, 0.09753782, 0.09478594,
       0.09357385, 0.09245105, 0.09265357, 0.09180266, 0.09190754,
       0.09219316, 0.09285381, 0.087317  , 0.08682802, 0.08435756,
       0.08129261, 0.07820942, 0.07523812, 0.07234206, 0.07302861,
       0.07495274, 0.07817484, 0.08032417, 0.08411147, 0.08686379,
       0.08966244, 0.09154433, 0.09137275, 0.08796845, 0.08838765,
       0.08273996, 0.07686574, 0.07195451, 0.06777752, 0.06497194,
       0.06476241],
        msi_mean=[902.3201592 , 1090.78438656,  989.37881814, 1686.55931541],
        sar_mean=[-19.619953, -27.59927],
    ),
    std=dict(
        hsi_std=[0.0338367 , 0.03383137, 0.03385848, 0.03390996, 0.03402681,
       0.03420128, 0.034422  , 0.03469419, 0.0348803 , 0.03507446,
       0.0355257 , 0.03567675, 0.03579371, 0.03596539, 0.03622309,
       0.03652683, 0.0369728 , 0.03735592, 0.03772956, 0.03816878,
       0.03864674, 0.03921149, 0.03985281, 0.04083455, 0.04115183,
       0.04144993, 0.04167945, 0.04190701, 0.0418512 , 0.04140075,
       0.0408082 , 0.04013697, 0.03959468, 0.03906678, 0.03862146,
       0.03852635, 0.03847326, 0.03877828, 0.03905807, 0.0389075 ,
       0.03770031, 0.03658479, 0.03549273, 0.03453911, 0.03312785,
       0.03173183, 0.03075266, 0.03008573, 0.02981177, 0.02980389,
       0.02998579, 0.03035695, 0.03078329, 0.0315527 , 0.03216608,
       0.03234724, 0.03220377, 0.03233009, 0.03266193, 0.03278077,
       0.03382322, 0.03414004, 0.03392119, 0.03359616, 0.03386534,
       0.03343628, 0.03181756, 0.03022669, 0.02864773, 0.02745449,
       0.02646621, 0.0261914 , 0.02624583, 0.02715282, 0.02930475,
       0.03200178, 0.03462014, 0.03576441, 0.0367125 , 0.03824908,
       0.0400046 , 0.04043352, 0.04005068, 0.0399938 , 0.03957121,
       0.03981536, 0.04006762, 0.04077112, 0.04088942, 0.0411583 ,
       0.04122896, 0.04138721, 0.03869136, 0.03842798, 0.03730686,
       0.03592587, 0.03457858, 0.03335235, 0.03224328, 0.03283242,
       0.03406012, 0.03578215, 0.03696036, 0.03875346, 0.03998185,
       0.0410889 , 0.04169787, 0.04159559, 0.04000605, 0.04016274,
       0.03759056, 0.03493045, 0.0327265 , 0.03090931, 0.02973136,
       0.02977929],
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
    bands=116,
    hidden_channels=[64, 32],
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
        share_net=False, # 各模态net是否权重共享
        net=dict(
            hsi=dict(type='VisionTransformer',
                    img_size=crop_size,
                    patch_size=16,
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
            msi=dict(type='VisionTransformer',
                    img_size=crop_size,
                    patch_size=16,
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
            sar=dict(type='VisionTransformer',
                    img_size=crop_size,
                    patch_size=16,
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
        ),
    ),
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
                 lr=1e-5, 
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
