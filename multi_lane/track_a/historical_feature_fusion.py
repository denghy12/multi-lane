"""Original feature fusion with the historical numerical and supervision recipe."""
from pathlib import Path
from .legacy_full_reference import historical_training_flags

ARMS = ('FULL_ASL', 'THREE_BCE_AUX', 'THREE_ASL_NO_AUX', 'THREE_ASL_AUX')


def modes(arm):
    if arm not in ARMS:
        raise ValueError('Unknown historical feature-fusion arm')
    return dict(view_fusion='disabled' if arm == 'FULL_ASL' else 'fixed_three_view',
                loss_routing='joint_bce' if arm == 'THREE_BCE_AUX' else 'adapter_asl',
                view_auxiliary_loss_weight=.1 if arm in {'THREE_BCE_AUX', 'THREE_ASL_AUX'} else 0.)


def arm_for(values):
    values = dict(values)
    values['loss_routing'] = values.get('loss_routing', values.get('parameter_group_loss_routing'))
    for arm in ARMS:
        if all(values.get(k) == v for k, v in modes(arm).items()):
            return arm
    raise ValueError('Not a declared historical feature-fusion arm')


COMMON = dict(seed=0, training_protocol='incremental', historical_feature_fusion=True,
    strong_full_asl=False, module_stacking=False, moe_paired_audit=False,
    fixed_view_paired_audit=False, protected_parax_paired_audit=False,
    calibration_fraction=0., workers=2, eval_workers=None, skip_validation_eval=False,
    also_report_test=False, paired_full_person=True, input_mode='full',
    input_normalization='clip', train_crop_scale=[.05, 1.], full_crop_mode='legacy',
    person_crop_margin=.15, person_color_jitter_strength=.1, person_color_jitter_probability=.2,
    selector_mode='shared', prompt_mode='shared', selector_conditioning='disabled',
    view_classifier_mode='shared_post_fusion', view_gradient_routing='joint',
    adapter_mode='image_token', adapter_bottleneck_dim=32, adapter_layer_indices=[1],
    adapter_residual_scale=.1, adapter_activation='relu', adapter_residual_gate_mode='fixed',
    adapter_view_mode='shared', adapter_view_bottleneck_dim=0, adapter_weight_decay=0.,
    adapter_regularization='none', parax_mode='disabled', cross_view_mode='disabled',
    spatial_summary_mode='disabled', joint_summary_mode='disabled',
    training_loss_mode='legacy_full_zero', weight_decay=0., supervised_loss_scale=1.,
    train_batch_size=64, eval_batch_size=64, amp_initial_scale=65536., amp_growth_interval=2000,
    save_compact_checkpoints=True, save_pre_evaluation_recovery=True)


def _check(values, expected):
    differences = {k: [v, values.get(k)] for k, v in expected.items() if values.get(k) != v}
    if differences:
        raise ValueError('Historical feature-fusion configuration differs: ' + str(differences))


def validate_arguments(args):
    values = dict(vars(args))
    for key in ('train_crop_scale', 'adapter_layer_indices'):
        values[key] = list(values[key])
    _check(values, {**COMMON, 'epochs': 30, 'source_learning_rate': .05,
        'source_reference_batch_size': 256, 'adapter_learning_rate': .0004,
        'adapter_task_init': 'independent', 'scheduler_mode': 'cosine',
        'scheduler_min_lr_ratio': 0., 'scheduler_warmup_ratio': 0.,
        'asl_gamma_neg': 9.8, 'asl_gamma_pos': 0., 'asl_clip': .05, 'asl_eps': 1e-8,
        'no_amp': False, 'no_tf32': False, 'crossfit_folds': 0, 'gradient_clip_norm': 0.})
    if not ((args.max_tasks == 8 and args.optimizer_updates_per_task is None and args.reporting_split == 'test')
            or (args.max_tasks == 2 and args.optimizer_updates_per_task == 2 and args.reporting_split == 'val')):
        raise ValueError('Use complete 8-task training or the declared 2-task smoke')
    arm_for(values)


def validate_config(config):
    expected = dict(COMMON)
    expected.pop('adapter_regularization')  # config stores its detailed policy separately
    _check(config, expected)
    _check(config, dict(learning_rate=.0125, optimizer='Adam_reset_per_task',
        adapter_learning_rate=.0004, adapter_task_initialization='independent',
        amp=True, tf32=True, fixed_view_sampler_protocol=None))
    return arm_for(config)


def training_arguments(args, root, arm):
    flags = historical_training_flags(args.data_root, args.clip_checkpoint,
                                       root / arm, smoke=args.smoke, seed=0)
    flags += ['--historical-feature-fusion', '--save-pre-evaluation-recovery']
    for key, value in modes(arm).items():
        flag = '--' + key.replace('_', '-')
        if flag in flags:
            flags[flags.index(flag) + 1] = str(value)
        else:
            flags += [flag, str(value)]
    if arm != 'FULL_ASL':
        flags += ['--face-manifest-root', str(args.face_manifest_root)]
    return list(map(str, flags))
