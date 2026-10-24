"""Locked September b32 recipe and bounded, one-factor parameter comparisons."""
from .historical_feature_fusion import COMMON, _check
from .legacy_full_reference import historical_training_flags

LEGACY_COMMIT = '36b77535cb2bc7f866a7950395ef14d26c1b4058'
PROFILES = {
    'THREE_BASE': dict(adapter_residual_scale=.03, adapter_learning_rate=.0004, view_auxiliary_loss_weight=.1),
    'THREE_SCALE_001': dict(adapter_residual_scale=.01),
    'THREE_SCALE_005': dict(adapter_residual_scale=.05),
    'THREE_LR_0002': dict(adapter_learning_rate=.0002),
    'THREE_LR_0008': dict(adapter_learning_rate=.0008),
    'THREE_AUX_0': dict(view_auxiliary_loss_weight=0.),
    'THREE_AUX_02': dict(view_auxiliary_loss_weight=.2),
    'FULL_BASE': dict(view_auxiliary_loss_weight=0.),
}


def recipe(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown b32 profile: ' + str(profile))
    return dict(PROFILES['THREE_BASE'], **PROFILES[profile]) if profile != 'THREE_BASE' else dict(PROFILES[profile])


def expected_common(seed, profile):
    if seed not in {0, 1, 3}:
        raise ValueError('This experiment declares seeds 0, 1, and 3')
    return dict(COMMON, seed=seed, skip_validation_eval=True,
        person_color_jitter_strength=0., person_color_jitter_probability=0.,
        historical_b32_profile=profile,
        view_fusion='disabled' if profile == 'FULL_BASE' else 'fixed_three_view',
        **{k: v for k, v in recipe(profile).items() if k != 'adapter_learning_rate'})


def validate_arguments(args):
    values = vars(args).copy()
    for key in ('train_crop_scale', 'adapter_layer_indices'):
        values[key] = list(values[key])
    _check(values, dict(expected_common(args.seed, args.historical_b32_profile),
        epochs=30, source_learning_rate=.05, source_reference_batch_size=256,
        adapter_learning_rate=recipe(args.historical_b32_profile)['adapter_learning_rate'],
        adapter_task_init='independent', scheduler_mode='cosine', scheduler_min_lr_ratio=0.,
        scheduler_warmup_ratio=0., asl_gamma_neg=9.8, asl_gamma_pos=0., asl_clip=.05,
        asl_eps=1e-8, no_amp=False, no_tf32=False, crossfit_folds=0,
        gradient_clip_norm=0., loss_routing='adapter_asl'))
    if not ((args.max_tasks == 8 and args.optimizer_updates_per_task is None and args.reporting_split == 'test')
            or (args.max_tasks == 2 and args.optimizer_updates_per_task == 2 and args.reporting_split == 'val')):
        raise ValueError('Use eight complete tasks or two-task/two-update smoke')
    if not args.historical_feature_fusion:
        raise ValueError('b32 profiles require historical feature-fusion numerics')


def validate_config(config):
    expected = expected_common(config['seed'], config['historical_b32_profile'])
    expected.pop('adapter_regularization')
    expected['eval_workers'] = 2
    _check(config, expected)
    _check(config, dict(learning_rate=.0125, optimizer='Adam_reset_per_task',
        adapter_learning_rate=recipe(config['historical_b32_profile'])['adapter_learning_rate'],
        adapter_task_initialization='independent', amp=True, tf32=True,
        fixed_view_sampler_protocol=None, parameter_group_loss_routing='adapter_asl',
        evaluation_scope='samples_intersect_seen_classes'))
    return config['historical_b32_profile']


def replace_flag(flags, key, value):
    key = '--' + key.replace('_', '-')
    if key in flags:
        flags[flags.index(key) + 1] = str(value)
    else:
        flags += [key, str(value)]


def training_arguments(args, root, seed, profile, native=False):
    # The old Full helper restricts seeds to 0/1/2; overwrite its CLI seed for
    # the newly authorized seed3 without changing that older audit's policy.
    flags = historical_training_flags(args.data_root, args.clip_checkpoint,
        root, smoke=args.smoke, seed=1 if seed == 3 else seed)
    for key, value in dict(seed=seed, person_color_jitter_strength=0.,
        person_color_jitter_probability=0., view_fusion='disabled' if profile == 'FULL_BASE' else 'fixed_three_view',
        view_classifier_mode='shared_post_fusion', view_gradient_routing='joint', **recipe(profile)).items():
        replace_flag(flags, key, value)
    flags += ['--skip-validation-eval']
    if profile != 'FULL_BASE':
        flags += ['--face-manifest-root', str(args.face_manifest_root), '--view-evaluation-diagnostics']
    if not native:
        flags += ['--historical-feature-fusion', '--historical-b32-profile', profile,
                  '--save-pre-evaluation-recovery']
    return list(map(str, flags))


def experiment_cells(smoke=False):
    if smoke:
        return [(1, 'THREE_BASE'), (1, 'FULL_BASE'), (3, 'THREE_AUX_0')]
    cells = [(s, p) for s in (1, 0, 3) for p in ('THREE_BASE', 'FULL_BASE')]
    cells += [(s, p) for p in PROFILES if p not in {'THREE_BASE', 'FULL_BASE'} for s in (0, 3)]
    return cells
