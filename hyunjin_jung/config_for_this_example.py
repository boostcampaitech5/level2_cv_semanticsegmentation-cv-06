

# Train Segformer Mit B0
_base_ = [
    "../../mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py",
    "dataset_setting.py",
    "../../mmsegmentation/configs/_base_/default_runtime.py"
]

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    size=(512, 512),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type='EncoderDecoderWithoutArgmax',
    init_cfg=dict(
        type='Pretrained',
        # load ADE20k pretrained EncoderDecoder from mmsegmentation
        checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth"
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='DepthwiseSeparableASPPHeadWithoutAccuracy',
        num_classes=29,
        loss_decode=dict(
            type='BCEDiceLoss'
        ),
    ),
    auxiliary_head=dict(
        type='FCNHeadWithoutAccuracy',
        num_classes=29,
        loss_decode=dict(
            type='BCEDiceLoss'
        ),
    ),
)

# optimizer
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=3000,
        end=80000,
        by_epoch=False,
    )
]
# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)
