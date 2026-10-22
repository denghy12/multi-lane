"""Strict, one-hop continuation of the locked spatial experiment at task end."""
import hashlib
import json
from pathlib import Path
import shutil

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_source_git(config, task):
    resume = config.get('resumed_from')
    if resume is not None and task < resume['boundary_task']:
        return resume['source_git']
    return config['git']


def validate_boundary(payload, current):
    old = payload['config']
    if (payload.get('schema_version') != 1 or payload.get('kind') != 'task_boundary_before_reporting'
            or payload.get('next_action') != 'report_completed_task_then_activate_next'
            or payload.get('source_git') != old['git'] or old['git'].get('dirty') is not False
            or current['git'].get('dirty') is not False or old.get('resumed_from')):
        raise ValueError('Boundary lacks clean, single-hop provenance')
    differences = [key for key in set(old) | set(current)
                   if key != 'git' and (bool(old.get(key)) != bool(current.get(key))
                   if key == 'module_stacking' else old.get(key) != current.get(key))]
    if differences:
        raise ValueError('Resume configuration differs: ' + ', '.join(sorted(differences)))
    if old.get('module_stacking'):
        from .module_stacking import validate_config
        validate_config(old)
    elif (old.get('adapter_mode') != 'disabled' or old.get('parax_mode') != 'disabled'
            or old.get('joint_summary_mode') != 'disabled' or old.get('cross_view_mode') != 'disabled'
            or old.get('selector_conditioning', 'disabled') != 'disabled'
            or old.get('view_fusion') != 'fixed_three_view'
            or old.get('view_classifier_mode') != 'shared_post_fusion'
            or old.get('selector_mode') != 'shared' or old.get('prompt_mode') != 'shared'
            or old.get('workers') != 0 or old.get('eval_workers') != 0
            or old.get('reporting_split') != 'val' or old.get('also_report_test')
            or not old.get('skip_validation_eval') or not old.get('moe_paired_audit')
            or not old.get('save_pre_evaluation_recovery')
            or old.get('spatial_summary_mode') not in ('disabled', 'absolute', 'target')):
        raise ValueError('Resume is limited to the original spatial experiment profile')
    task = payload['task_id']
    if not 0 <= task < old['max_tasks']:
        raise ValueError('Boundary task is outside the run')
    provenance = payload['provenance']
    if set(provenance['training_history']) != {str(t) for t in range(task + 1)}:
        raise ValueError('Boundary history is incomplete')
    for rows in provenance['training_history'].values():
        if not rows or any(row.get('skipped_optimizer_steps', 0) != 0 for row in rows):
            raise ValueError('Boundary has incomplete or skipped training')
        if old.get('optimizer_updates_per_task') is not None:
            if sum(row['optimizer_steps'] for row in rows) != old['optimizer_updates_per_task']:
                raise ValueError('Boundary has an incomplete update budget')
        elif len(rows) != old['epochs_per_task']:
            raise ValueError('Boundary has an incomplete epoch budget')
    if [r['task_id'] for r in provenance['completed_reporting']] != list(range(task)):
        raise ValueError('Boundary reporting prefix is incomplete')
    if payload['sampler_generator'] is None or not payload['rng']['torch_cuda']:
        raise ValueError('Boundary lacks sampler/CUDA RNG state')
    if not all(k in payload['training_state'] for k in ('optimizer', 'scheduler', 'scaler', 'optimizer_parameter_names')):
        raise ValueError('Boundary lacks complete optimizer provenance')


def prepare_boundary(path, config, output):
    path, output = Path(path), Path(output)
    before = sha256(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_boundary(payload, config)
    source = path.parent.parent
    if json.loads((source / 'config.json').read_text()) != payload['config']:
        raise ValueError('Source config changed after the snapshot')
    task = payload['task_id']
    config['resumed_from'] = dict(boundary_task=task, source_run=str(source),
        boundary_checkpoint=str(path), boundary_sha256=before, source_git=payload['source_git'],
        inherited_updates=sum(r['optimizer_steps'] for rows in payload['provenance']['training_history'].values() for r in rows),
        first_new_training_task=task + 1, optimizer_policy='reset_per_new_task_as_original',
        rng_policy='restore_pre_reporting_then_replay_reporting')
    inherited = {}
    for t in range(task + 1):
        folders = [('recovery_checkpoints', f'task{t}_before_reporting.pth'),
                   ('calibration_split_provenance', f'task{t}.json')]
        if t < task:
            folders += [(folder, f'task{t}.{suffix}') for folder, suffix in
                        [('compact_checkpoints', 'pth'), ('val_scores', 'npz'), ('calibration_scores', 'npz')]]
        for folder, name in folders:
            src, dst = source / folder / name, output / folder / name
            if dst.exists():
                raise FileExistsError(dst)
            value = sha256(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha256(src) != value or sha256(dst) != value:
                raise RuntimeError('Inherited artifact changed during copying')
            inherited[str(dst.relative_to(output))] = dict(source=str(src), sha256=value)
    rows = json.loads((source / 'calibration_metrics.json').read_text()) if task else []
    payload['inherited_calibration_rows'] = [r for r in rows if r['task_id'] < task]
    if len(payload['inherited_calibration_rows']) != task:
        raise ValueError('Source holdout reporting prefix is incomplete')
    if sha256(path) != before:
        raise RuntimeError('Source boundary changed while preparing resume')
    (output / 'resume_manifest.json').write_text(json.dumps(dict(
        status='prepared', **config['resumed_from'], inherited_files=inherited,
        execution_git=config['git'], source_unchanged=True), indent=2) + '\n')
    return payload


def restore_boundary_model(model, payload):
    result = model.load_state_dict(payload['model'], strict=False)
    if result.unexpected_keys or not result.missing_keys or any(
            not name.startswith('visual_encoder.') for name in result.missing_keys):
        raise ValueError('Boundary does not contain the complete method state')
    model.restore_task(payload['task_id'])
    flags = payload['requires_grad']
    if set(flags) != dict(model.named_parameters()).keys():
        raise ValueError('Boundary trainability names differ')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(flags[name])
    model.train(payload['model_training'])
