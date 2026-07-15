_base_ = './grounding_dino_swin_t.py'

data_root = '../data/cityscapes_coco/'
class_names = (
    'bicycle',
    'building',
    'bus',
    'car',
    'fence',
    'motorcycle',
    'person',
    'pole',
    'rider',
    'road',
    'sidewalk',
    'sky',
    'terrain',
    'traffic light',
    'traffic sign',
    'train',
    'truck',
    'vegetation',
    'wall',
)

metainfo = dict(classes=class_names)

# Detection-only finetuning. This disables the LLM/LLaVA branch and keeps only
# the GroundingDINO bbox training path.
model = dict(
    lmm=None,
    lmm_connector=None,
    lmm_region_loss_weight=0.0,
    lmm_image_loss_weight=0.0,
    use_lmm_cross_attn=False,
    use_image_level_cross_attn=False,
    num_lmm_new_layers=0,
    num_region_caption=0)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

cityscapes_train_dataset = dict(
    type='ODVGDataset',
    data_root=data_root,
    ann_file='train/odvg_train.jsonl',
    label_map_file='odvg_label_map.json',
    data_prefix=dict(img='train'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=train_pipeline,
    return_classes=True,
    actual_dataset_mode='OD',
    use_short_cap=False,
    use_uniform_prompt=True,
    clean_caption=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=cityscapes_train_dataset)

base_test_pipeline = _base_.test_pipeline
base_test_pipeline[-1]['meta_keys'] = (
    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
    'text', 'custom_entities', 'tokens_positive')

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid'),
        test_mode=True,
        pipeline=base_test_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox')
test_evaluator = val_evaluator

# Cityscapes is denser than the mine-scene split, so use a longer default run.
max_iter = 30000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=5000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[20000, 27000],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=5, save_best='coco/bbox_mAP'),
    visualization=dict(type='GroundingVisualizationHook'),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)

auto_scale_lr = dict(base_batch_size=16, enable=False)
