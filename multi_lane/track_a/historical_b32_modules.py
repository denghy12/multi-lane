"""Seed1 summary-module additions to the exactly reproduced historical b32 recipe."""
from .historical_b32_search import expected_common, training_arguments as base_arguments, replace_flag
from .historical_feature_fusion import _check

ARMS = ('BASE', 'ATTENTION', 'ATTENTION_BOUND', 'POSITION', 'POSITION_CALIBRATED')


def modes(arm):
    if arm not in ARMS:
        raise ValueError('Unknown historical b32 summary module')
    return dict(cross_view_mode='task_full' if arm.startswith('ATTENTION') else 'disabled',
        spatial_summary_mode='target' if arm.startswith('POSITION') else 'disabled',
        cross_view_residual_cap=.1 if arm == 'ATTENTION_BOUND' else 0.,
        spatial_summary_bias_mode='content_relative' if arm == 'POSITION_CALIBRATED' else 'raw')


def expected(arm, private_selectors=False):
    result = expected_common(1, 'THREE_BASE')
    result.update(historical_b32_profile=None, historical_b32_modules=True,
        historical_b32_module_profile=arm, cross_view_layer=5, cross_view_dim=64,
        cross_view_heads=4, cross_view_scale=.1, **modes(arm))
    if private_selectors:
        if arm != 'ATTENTION_BOUND':
            raise ValueError('Independent view Selectors require bounded dual-view attention')
        result['selector_mode'] = 'view_specific'
    return result


def validate_arguments(args):
    values = vars(args).copy()
    for k in ('adapter_layer_indices', 'train_crop_scale'):
        values[k] = list(values[k])
    private = getattr(args, 'historical_b32_private_selectors', False)
    if private and not args.cross_view_face_attention:
        raise ValueError('Independent Selectors are declared only for the dual attention control')
    _check(values, dict(expected(args.historical_b32_module_profile, private),
        epochs=30, source_learning_rate=.05, source_reference_batch_size=256,
        adapter_learning_rate=.0004, view_fusion_learning_rate=.0004,
        adapter_task_init='independent', scheduler_mode='cosine',
        scheduler_min_lr_ratio=0., scheduler_warmup_ratio=0.,
        asl_gamma_neg=9.8, asl_gamma_pos=0., asl_clip=.05, asl_eps=1e-8,
        no_amp=False, no_tf32=False, loss_routing='adapter_asl', crossfit_folds=0,
        gradient_clip_norm=0.))
    if not ((args.max_tasks == 8 and args.optimizer_updates_per_task is None and args.reporting_split == 'test')
            or (args.max_tasks == 2 and args.optimizer_updates_per_task == 2 and args.reporting_split == 'val')):
        raise ValueError('Use full eight tasks or the two-task smoke')


def validate_config(config):
    arm = config['historical_b32_module_profile']
    private = config.get('historical_b32_private_selectors', False)
    if private and not config.get('cross_view_face_attention', False):
        raise ValueError('Independent Selector provenance lacks reliable Face attention')
    values = expected(arm, private)
    values.pop('adapter_regularization'); values['eval_workers'] = 2
    if modes(arm)['cross_view_mode'] == 'disabled':
        for k in ('cross_view_layer', 'cross_view_dim', 'cross_view_heads', 'cross_view_scale'):
            values.pop(k)
    _check(config, values)
    _check(config, dict(learning_rate=.0125, adapter_learning_rate=.0004,
        view_fusion_learning_rate=None if arm == 'BASE' else .0004,
        parameter_group_loss_routing='adapter_asl', adapter_parameter_objective='asl',
        optimizer='Adam_reset_per_task', adapter_task_initialization='independent',
        amp=True, tf32=True, fixed_view_sampler_protocol=None,
        evaluation_scope='samples_intersect_seen_classes'))
    return arm


def training_arguments(args, root, arm):
    flags = base_arguments(args, root, 1, 'THREE_BASE')
    i = flags.index('--historical-b32-profile'); del flags[i:i+2]
    flags += ['--historical-b32-modules', '--historical-b32-module-profile', arm]
    for k, v in dict(cross_view_layer=5, cross_view_dim=64, cross_view_heads=4,
            cross_view_scale=.1, view_fusion_learning_rate=.0004, **modes(arm)).items():
        replace_flag(flags, k, v)
    return flags
