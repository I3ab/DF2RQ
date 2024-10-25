_base_ = [
    '../_base_/datasets/c2seg_bw.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
img_norm_cfg = dict(
    mean=dict(
        hsi_mean=[0.1001394 , 0.09870405, 0.09726687, 0.09571745, 0.09436516,
       0.09316147, 0.09209691, 0.09150705, 0.09123235, 0.09159759,
       0.0926962 , 0.09354472, 0.09417944, 0.09498482, 0.09591238,
       0.09687698, 0.09794021, 0.09875634, 0.09944061, 0.10001577,
       0.10057223, 0.10106092, 0.10147923, 0.1025512 , 0.10221649,
       0.10169423, 0.10123207, 0.10127601, 0.10128875, 0.10117025,
       0.10181322, 0.10330562, 0.10573918, 0.10862102, 0.11189609,
       0.11571678, 0.1191404 , 0.12329105, 0.1265122 , 0.12831034,
       0.12730566, 0.12605366, 0.12435719, 0.12276398, 0.12011122,
       0.11754964, 0.1159204 , 0.11505324, 0.11481663, 0.11501008,
       0.1156723 , 0.11612691, 0.11643037, 0.11783747, 0.11812072,
       0.11735038, 0.11597074, 0.1154554 , 0.11535717, 0.11465836,
       0.11643455, 0.11757086, 0.11837251, 0.11951654, 0.12238225,
       0.12357461, 0.12228527, 0.12034021, 0.11815685, 0.11613433,
       0.11333787, 0.11204874, 0.111844  , 0.11313384, 0.11703596,
       0.12280751, 0.12856954, 0.12955247, 0.12934534, 0.12974162,
       0.13056066, 0.12748828, 0.12257755, 0.11783736, 0.11234815,
       0.10804591, 0.10463075, 0.10372481, 0.10279851, 0.10336311,
       0.10444809, 0.10635371, 0.1014701 , 0.09963221, 0.09599006,
       0.0922062 , 0.08906296, 0.08550005, 0.0826054 , 0.08261744,
       0.08387533, 0.08648831, 0.08848102, 0.09177398, 0.09515489,
       0.09838553, 0.10020875, 0.10037063, 0.09749393, 0.09769067,
       0.09267425, 0.08795069, 0.08422044, 0.08081194, 0.07866203,
       0.07824473],
        msi_mean=[632.61231547, 796.31124089, 837.08928824, 1384.07002065],
        sar_mean=[-17.09582294, -25.2831639 ],
    ),
    std=dict(
        hsi_std=[0.02889844, 0.02900737, 0.02917334, 0.02936207, 0.02962804,
       0.02988079, 0.03016525, 0.03050014, 0.03077815, 0.03104836,
       0.0314112 , 0.03162646, 0.03182289, 0.03207011, 0.03233328,
       0.03267017, 0.03311934, 0.03353089, 0.03392887, 0.03434746,
       0.03478206, 0.03522748, 0.03577327, 0.0365742 , 0.03697609,
       0.03737775, 0.03770849, 0.03800497, 0.03796539, 0.03752053,
       0.03684168, 0.03612136, 0.03553707, 0.03513617, 0.03510458,
       0.03561947, 0.03633359, 0.03744876, 0.03850303, 0.03911686,
       0.03888935, 0.03850957, 0.0380168 , 0.0375475 , 0.03671327,
       0.03595005, 0.03554788, 0.03540175, 0.03543251, 0.03557283,
       0.0358862 , 0.03620649, 0.03638202, 0.03682773, 0.03701171,
       0.03685154, 0.03657245, 0.03656103, 0.03665257, 0.03653765,
       0.03705241, 0.03737051, 0.0375204 , 0.03771046, 0.03860869,
       0.03884868, 0.038094  , 0.0371543 , 0.03615885, 0.03526594,
       0.03424253, 0.03377355, 0.03369515, 0.03409358, 0.03540608,
       0.03734946, 0.03911084, 0.03958997, 0.03975371, 0.04021965,
       0.04100841, 0.04041042, 0.03922197, 0.03839266, 0.03754245,
       0.03720096, 0.03708465, 0.03760697, 0.03783601, 0.03824569,
       0.03862916, 0.03909192, 0.03677236, 0.03593542, 0.03447484,
       0.03270573, 0.03113155, 0.02967963, 0.02850913, 0.02864176,
       0.02936778, 0.03050933, 0.0313825 , 0.03269601, 0.03376372,
       0.03479154, 0.0354281 , 0.03552333, 0.03416419, 0.03404794,
       0.03179888, 0.02944793, 0.02762477, 0.02612824, 0.02503884,
       0.02478187],
        msi_std=[331.63181692, 397.48872929, 447.40186075, 618.9137618 ],
        sar_std=[4.84774285, 4.43630586],
    )
)
crop_size = (256, 256)
data_preprocessor = dict(
    type='MultimodalSegDataPreProcessor',
    size=crop_size,
    norm_cfg = img_norm_cfg,
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
    hidden_channels=[32],
    kernel_size=[1])
sar_stem = dict(
    bands=2,
    hidden_channels=[32],
    kernel_size=[1])

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MultibranchBackbone',
        bands=dict(hsi=116, msi=4, sar=2),
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
