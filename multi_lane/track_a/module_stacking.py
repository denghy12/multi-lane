"""Locked factorial protocol. No torch import: safe in the job controller."""
ARMS = ('F000', 'F100', 'F010', 'F001', 'F110', 'F101', 'F011', 'F111')
STAGES = (ARMS[:4], ARMS[4:7], ARMS[7:])


def modes(arm):
    if arm not in ARMS:
        raise ValueError('Unknown factorial arm')
    a, b, c = (x == '1' for x in arm[1:])
    return dict(adapter_mode='image_token' if a else 'disabled',
                cross_view_mode='task_full' if b else 'disabled',
                spatial_summary_mode='target' if c else 'disabled')


def arm_for(config):
    for arm in ARMS:
        if all(config.get(k) == v for k, v in modes(arm).items()):
            return arm
    raise ValueError('Not a locked factorial combination')


def _check(values, expected):
    bad = [k for k, v in expected.items() if values.get(k) != v]
    if bad:
        raise ValueError('Stacking protocol mismatch: ' + ', '.join(bad))


COMMON = dict(seed=0, module_stacking=True, moe_paired_audit=True,
              calibration_fraction=.1, view_auxiliary_loss_weight=.1,
              view_fusion='fixed_three_view', view_classifier_mode='shared_post_fusion',
              selector_mode='shared', prompt_mode='shared', selector_conditioning='disabled',
              parax_mode='disabled', joint_summary_mode='disabled', workers=0,
              eval_workers=0, reporting_split='val', also_report_test=False,
              skip_validation_eval=True, save_pre_evaluation_recovery=True,
              save_compact_checkpoints=True, num_selectors=10, num_prompts=10,
              num_prompt_layers=5, cross_view_layer=5, cross_view_dim=64,
              cross_view_heads=4, cross_view_scale=.1,
              adapter_bottleneck_dim=32, adapter_layer_indices=[1],
              adapter_residual_scale=.1, adapter_activation='relu',
              adapter_view_bottleneck_dim=0, adapter_view_mode='shared',
              adapter_residual_gate_mode='fixed', adapter_regularization='none')


def validate_arguments(args):
    values = dict(vars(args)); values['adapter_layer_indices'] = list(args.adapter_layer_indices)
    values.update(num_prompts=10, num_prompt_layers=5)  # runner's fixed architecture
    _check(values, {**COMMON, 'training_protocol': 'incremental', 'loss_routing': 'joint_bce',
                    'view_gradient_routing': 'joint', 'epochs': 30, 'train_batch_size': 64,
                    'eval_batch_size': 64, 'adapter_task_init': 'independent',
                    'adapter_learning_rate': .0004, 'view_fusion_learning_rate': .0004,
                    'source_learning_rate': .05, 'source_reference_batch_size': 256,
                    'weight_decay': 0., 'scheduler_mode': 'cosine',
                    'scheduler_min_lr_ratio': 0., 'scheduler_warmup_ratio': 0.,
                    'training_loss_mode': 'legacy_full_zero', 'supervised_loss_scale': 1.,
                    'full_crop_mode': 'legacy', 'person_transform_mode': 'letterbox',
                    'person_crop_margin': .15, 'amp_initial_scale': 1024.,
                    'amp_growth_interval': 1000000000})
    arm_for(values)


def validate_config(config):
    # Inactive Adapter LR is stored as null; the controller still passes the
    # same locked values to every arm. Training validates the full arguments.
    expected = dict(COMMON)
    expected.pop('adapter_regularization')
    _check(config, expected)
    _check(config, dict(parameter_group_loss_routing='joint_bce',
                        train_batch_size=64, eval_batch_size=64,
                        learning_rate=.0125, weight_decay=0.,
                        optimizer='Adam_reset_per_task',
                        adapter_task_initialization='independent'))
    arm_for(config)


def runtime_bypass(model, disabled):
    disabled = set(disabled)
    if not model.module_stacking or disabled - {'A', 'B', 'C'}:
        raise ValueError('Invalid factorial runtime bypass')
    if model.adapter_bank is not None:
        model.set_adapter_runtime_enabled('A' not in disabled)
    model.cross_view_runtime_enabled = model.cross_view_bank is not None and 'B' not in disabled
    model.spatial_summary_runtime_enabled = model.spatial_summary_bank is not None and 'C' not in disabled


def factorial_contrasts(values):
    if set(values) != set(ARMS):
        raise ValueError('All eight cells are required for factorial contrasts')
    m = lambda s: values['F' + s]
    return dict(
        A_given_BC=m('111')-m('011'), B_given_AC=m('111')-m('101'), C_given_AB=m('111')-m('110'),
        AB_given_C0=m('110')-m('100')-m('010')+m('000'),
        AC_given_B0=m('101')-m('100')-m('001')+m('000'),
        BC_given_A0=m('011')-m('010')-m('001')+m('000'),
        AB_given_C1=m('111')-m('101')-m('011')+m('001'),
        AC_given_B1=m('111')-m('110')-m('011')+m('010'),
        BC_given_A1=m('111')-m('110')-m('101')+m('100'),
        ABC=m('111')-m('110')-m('101')-m('011')+m('100')+m('010')+m('001')-m('000'))
