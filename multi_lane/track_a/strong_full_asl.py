"""Locked full-TRAIN, aux0 transfer protocol. No Torch import in controller."""
ARMS = ('S_BCE', 'S_ASL', 'T_BCE', 'T_ASL', 'T_ASL_ATTN', 'T_ASL_POS', 'T_ASL_BOTH')
LEGACY_COMMIT = 'f1ada2291ae40aaf17416ae96d6431505f2bc575'


def modes(arm):
    if arm not in ARMS:
        raise ValueError('Unknown strong Full arm')
    return dict(adapter_mode='image_token',
                view_fusion='disabled' if arm.startswith('S_') else 'fixed_three_view',
                cross_view_mode='task_full' if arm.endswith(('ATTN', 'BOTH')) else 'disabled',
                spatial_summary_mode='target' if arm.endswith(('POS', 'BOTH')) else 'disabled',
                loss_routing='joint_bce' if arm.endswith('BCE') else 'adapter_asl')


def arm_for(config):
    values = dict(config)
    values['loss_routing'] = values.get('loss_routing', values.get('parameter_group_loss_routing'))
    for arm in ARMS:
        if all(values.get(k) == v for k,v in modes(arm).items()):
            return arm
    raise ValueError('Not a declared strong Full arm')


COMMON = dict(strong_full_asl=True, module_stacking=False, moe_paired_audit=False,
    seed=0, training_protocol='incremental', calibration_fraction=0.,
    view_auxiliary_loss_weight=0., view_gradient_routing='joint',
    view_classifier_mode='shared_post_fusion', selector_mode='shared', prompt_mode='shared',
    selector_conditioning='disabled', parax_mode='disabled', joint_summary_mode='disabled',
    workers=0, eval_workers=0, reporting_split='val', also_report_test=False,
    skip_validation_eval=True, save_pre_evaluation_recovery=True, save_compact_checkpoints=True,
    num_selectors=10, adapter_bottleneck_dim=32, adapter_layer_indices=[1],
    adapter_view_mode='shared', adapter_view_bottleneck_dim=0,
    adapter_activation='relu', adapter_residual_scale=.1, adapter_residual_gate_mode='fixed',
    cross_view_layer=5, cross_view_dim=64, cross_view_heads=4, cross_view_scale=.1,
    amp_initial_scale=1024., amp_growth_interval=1000000000,
    train_batch_size=64, eval_batch_size=64, full_crop_mode='legacy',
    person_crop_margin=.15, train_crop_scale=[.05, 1.], input_normalization='clip',
    weight_decay=0., training_loss_mode='legacy_full_zero')


def check(values, expected):
    bad = [k for k,v in expected.items() if values.get(k) != v]
    if bad:
        raise ValueError('Strong Full profile mismatch: '+', '.join(bad))


def validate_arguments(args):
    values = dict(vars(args))
    for key in ('adapter_layer_indices', 'train_crop_scale'):
        values[key] = list(values[key])
    check(values, {**COMMON, 'epochs':30, 'adapter_task_init':'independent',
        'adapter_learning_rate':.0004, 'view_fusion_learning_rate':.0004,
        'adapter_weight_decay':0., 'adapter_regularization':'none',
        'source_learning_rate':.05, 'source_reference_batch_size':256,
        'scheduler_mode':'cosine', 'scheduler_min_lr_ratio':0., 'scheduler_warmup_ratio':0.,
        'asl_gamma_neg':9.8, 'asl_gamma_pos':0., 'asl_clip':.05, 'asl_eps':1e-8,
        'person_transform_mode':'letterbox', 'person_color_jitter_strength':0.,
        'person_color_jitter_probability':0., 'supervised_loss_scale':1.,
        'crossfit_folds':0, 'gradient_clip_norm':0.})
    arm_for(values)


def validate_config(config):
    expected=dict(COMMON)
    for key in ('cross_view_layer','cross_view_dim','cross_view_heads','cross_view_scale'):
        expected.pop(key)  # inactive attention stores nullable architecture fields
    check(config, expected)
    check(config, dict(learning_rate=.0125, optimizer='Adam_reset_per_task',
        scheduler='CosineAnnealingLR_reset_per_task', epochs_per_task=30,
        adapter_learning_rate=.0004, adapter_weight_decay=0.,
        adapter_task_initialization='independent', adapter_parameter_objective=(
            'bce' if config['parameter_group_loss_routing']=='joint_bce' else 'asl')))
    arm_for(config)
