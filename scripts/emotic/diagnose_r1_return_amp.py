"""Read-only gradient scan at a saved whole-model boundary; never optimizer.step.

Fixed-parameter scan of the next epoch's TRAIN batches. This does not reproduce
parameter changes within the failed original epoch and cannot guarantee that a
subsequent resumed trajectory is finite.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a import runner
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import attach_evidence, restore_rng, rng_state
from multi_lane.track_a.r1_crossvit_experiment import model_from_config, sha, write
from scripts.emotic.run_r1_frozen_cross_view import input_paths
from scripts.emotic.run_r1_single_trajectory_fusion import assert_same


def digest_tensors(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def source_from_config(c, inputs):
    transform, _ = runner.build_transforms(c['input_normalization'], c['train_crop_scale'],
        input_mode=c['input_mode'], person_transform_mode=c['person_transform_mode'],
        person_color_jitter_strength=c['person_color_jitter_strength'],
        person_color_jitter_probability=c['person_color_jitter_probability'])
    data = EMOTIC(str(inputs.data_root.parent), train=True, transform=transform,
        input_mode=c['input_mode'], person_crop_margin=c['person_crop_margin'],
        face_manifest_root=inputs.face_manifest_root if c['input_mode']=='face_crop' else None)
    if c['r1_kind'] != 'native':
        data = attach_evidence(data, c['r1_receiver'], inputs.data_root.parent, inputs.face_manifest_root)
    return data


def scan(checkpoint, output, view='person', batch_limit=None):
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required to compare AMP with FP32')
    output.mkdir(parents=True)
    torch.set_num_threads(72)
    runner.set_seed(1, True)
    before_sha = sha(checkpoint)
    whole = torch.load(checkpoint, map_location='cpu')
    if whole['condition'] != 'full_extend':
        raise ValueError('Only the matched full-TRAIN control is accepted')
    saved = whole['branches'][view]
    training = saved['training']
    if any(r['skipped_optimizer_steps'] for r in training['history']):
        raise ValueError('Diagnostic boundary contains skipped updates')
    task = int(saved['task']); c = saved['config']; inputs = input_paths()
    expected = dict(training_loss_mode='legacy_full_zero', parameter_group_loss_routing='adapter_asl',
        train_batch_size=64, workers=2, temperature=1., adapter_regularization='none',
        adapter_regularization_fraction=0.)
    if any(c.get(k) != value for k, value in expected.items()):
        raise ValueError('Diagnostic recipe differs from the failed control')
    if any(c.get('asl', {}).get(k) != value for k, value in
           dict(gamma_neg=9.8, gamma_pos=0., clip=.05, eps=1e-8).items()):
        raise ValueError('Diagnostic ASL recipe differs')
    model = model_from_config(c, inputs.clip_checkpoint)
    model.load_own(saved)
    model.train()
    model_params, adapter_params, groups = runner.build_optimizer_groups(
        model, c['weight_decay'], adapter_learning_rate=c['adapter_learning_rate'],
        adapter_weight_decay=c['adapter_weight_decay'])
    opt = torch.optim.Adam(groups, lr=.0125)
    names = {id(p): n for n, p in model.named_parameters()}
    actual_names = [[names[id(p)] for p in group['params']] for group in opt.param_groups]
    if actual_names != training['optimizer_parameter_names']:
        raise AssertionError('Optimizer parameter layout differs')
    opt.load_state_dict(training['optimizer'])
    model_digest = digest_tensors(model.state_dict())
    source = source_from_config(c, inputs)
    indices, _ = runner.fit_calibration_indices(source, runner.task_indices(task), 0.)
    if view == 'face':
        indices = runner.filter_face_training_indices(source, indices)
    data = runner.LabelView(source, indices, runner.task_indices(task), include_sample_id=True)
    loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=False)
    current = list(runner.task_indices(task))
    modes = [('amp4096', True, 4096.), ('amp32', True, 32.), ('fp32', False, 1.)]
    scalers = {name: torch.cuda.amp.GradScaler(enabled=amp, init_scale=scale)
               for name, amp, scale in modes}
    result = {'source_checkpoint': str(checkpoint), 'source_sha256': before_sha,
        'task': task, 'completed_epoch': training['next_epoch'], 'view': view,
        'model_parameter_updates': 0, 'optimizer_steps': 0,
        'scan_type': 'fixed_saved_parameters_same_next_epoch_batches_and_forward_rng',
        'limitation': 'No within-epoch parameter updates: not a replay of the failed epoch trajectory',
        'train_people': len(data), 'expected_batches': len(loader), 'batch_limit': batch_limit,
        'source_scaler': training['scaler'], 'optimizer_restored': True,
        'rows': [], 'status': 'running'}
    write(output/'diagnostic.json', result)
    restore_rng(saved['rng'])
    for index, (images, targets, ids) in enumerate(loader):
        if batch_limit is not None and index >= batch_limit:
            break
        images = move_model_inputs(images, torch.device('cuda'))
        targets = targets.cuda(non_blocking=True).float()
        batch_rng = rng_state(); next_rng = None
        row = {'batch': index, 'people': len(ids), 'ids': list(ids), 'modes': {}}
        for name, amp, scale in modes:
            restore_rng(batch_rng)
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.current_all_logits(images)
                bce = runner.compute_training_loss(logits, targets, current, 1., 'legacy_full_zero')
                asl = runner.compute_asymmetric_training_loss(logits, targets, current, 1.,
                    'legacy_full_zero', gamma_neg=9.8, gamma_pos=0., clip=.05, eps=1e-8)
            runner.backward_routed_training_losses(bce, asl, model_params, adapter_params,
                scalers[name], False)
            bad, maximum = [], 0.
            for param in (*model_params, *adapter_params):
                grad = param.grad
                if grad is None:
                    raise AssertionError(f'Missing gradient: {names[id(param)]}')
                if not torch.isfinite(grad).all():
                    bad.append(names[id(param)])
                else:
                    maximum = max(maximum, float(grad.detach().abs().max()) / scale)
            row['modes'][name] = {'bce': float(bce.detach()), 'asl': float(asl.detach()),
                'logits_finite': bool(torch.isfinite(logits).all()),
                'losses_finite': bool(torch.isfinite(bce)&torch.isfinite(asl)),
                'gradient_finite': not bad, 'nonfinite_parameter_names': bad,
                'max_abs_unscaled_finite_gradient': maximum}
            if next_rng is None:
                next_rng = rng_state()
            del logits, bce, asl
        restore_rng(next_rng)
        result['rows'].append(row)
        write(output/'diagnostic.json', result)
        print('GRADIENT_SCAN', index, {m: v['gradient_finite'] for m,v in row['modes'].items()}, flush=True)
    model.zero_grad(set_to_none=True)
    if digest_tensors(model.state_dict()) != model_digest:
        raise AssertionError('Diagnostic mutated model parameters or buffers')
    # Adam tensors may be moved to device by load_state_dict; compare on CPU.
    current_opt = opt.state_dict()
    for state in current_opt['state'].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()
    assert_same(current_opt, training['optimizer'], 'read_only_optimizer')
    if sha(checkpoint) != before_sha:
        raise AssertionError('Source checkpoint changed')
    result.update(status='complete', scanned_batches=len(result['rows']),
        all_expected_batches_scanned=len(result['rows']) == len(loader),
        model_unchanged=True, optimizer_unchanged=True, source_sha_unchanged=True,
        mode_nonfinite_batches={name: sum(not x['modes'][name]['gradient_finite'] or
            not x['modes'][name]['losses_finite'] or not x['modes'][name]['logits_finite']
            for x in result['rows']) for name,_,_ in modes})
    write(output/'diagnostic.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--view', choices=('full','person','face'), default='person')
    p.add_argument('--batch-limit', type=int)
    a=p.parse_args()
    scan(a.checkpoint, a.output, a.view, a.batch_limit)


if __name__=='__main__':
    main()
