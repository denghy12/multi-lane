"""Reproduce the locked A and C readouts on the Task10 evaluation caches."""
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_tcp_probe import TASKS, gate_scores, identify_source, input_tensors, read
from scripts.emotic.fit_d9_task10_probe import group_weights, load_arrays, mix, scores


def verify_metrics(fused, targets, reliable, row):
    masks = {'all': np.ones(len(targets), dtype=bool), 'reliable': reliable, 'unreliable': ~reliable}
    for group, mask in masks.items():
        actual = scores(fused[mask], targets[mask])
        expected = row['metrics'][group]
        if actual is None or actual['people'] != expected['people'] or actual['positive_counts'] != expected['positive_counts']:
            raise ValueError(f'Evaluation population differs for {group}')
        for key in ('per_class_ap', 'mAP', 'bce'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-7,
                                       err_msg=f'Cannot reproduce {group} {key}')


@torch.inference_mode()
def replay(prior: Path, tcp: Path, task: int, split: str, source_mode="sample_holdout"):
    if task not in TASKS or split not in ('val', 'test'):
        raise ValueError('Only tasks 0, 3 and 6 on Validation or Test are available')
    torch.set_num_threads(1)
    root = tcp / 'probe'
    lock_path = root / 'tcp_validation_lock.json'
    lock = read(lock_path)
    protocol = lock['protocol']
    if source_mode not in ('sample_full', 'sample_holdout') or protocol.get('source_mode', 'sample_holdout') != source_mode:
        raise ValueError('TCP lock uses a different branch/fusion data protocol')
    expected_tasks = [6] if source_mode == 'sample_full' else list(TASKS)
    if (lock['selection_split'] != 'val' or protocol['seed'] != 1 or protocol['steps'] != 300 or
            task not in expected_tasks or protocol['tasks'] != expected_tasks or protocol['tcp_mse_coefficient'] != 1.):
        raise ValueError('Unexpected seed1 formal TCP Validation lock')
    for name, expected in lock['artifact_hashes'].items():
        if sha256_file(root/name) != expected:
            raise ValueError(f'Damaged TCP fit/Validation artifact: {name}')

    val_source = lock['train_val_sources'][str(task)]['val']
    source = val_source
    lock_hash = sha256_file(lock_path)
    if split == 'test':
        tested = read(root/'tcp_test_results.json')
        if tested['validation_lock_sha256'] != lock_hash or tested['protocol'] != protocol:
            raise ValueError('Test result references different fusion parameters')
        source = tested['test_sources'][str(task)]
    archive, identity = identify_source(prior, task, split, source_mode=source_mode)
    if (identity != source or any(identity[key] != val_source[key] for key in ('checkpoint_sha256', 'config_sha256'))):
        raise ValueError('Split or checkpoint differs from the sealed source')
    model_path = root / f'tcp_model_task{task}.json'
    model = read(model_path)
    if model['task'] != task or model_path.name not in lock['artifact_hashes']:
        raise ValueError('Unexpected source fusion model')
    report_path = root / f'tcp_{split}_task{task}.json'
    report = read(report_path)
    data = load_arrays(archive, task)
    x, valid, z, _, _ = input_tensors(data, model['feature_stats'])
    result = dict(ids=data['sample_ids'], y=data['targets'], z=z.numpy(), reliable=valid.numpy())
    fixed_logits = torch.as_tensor(data['fixed_logits'], dtype=torch.float32)
    verify_metrics(fixed_logits, data['targets'], data['reliable'], report['fixed'])
    result.update(logits_fixed=fixed_logits.numpy(), ap_fixed=np.asarray(report['fixed']['metrics']['all']['per_class_ap']))
    for method, suffix in [('A', 'a'), ('C_tcp', 'c')]:
        params = {k: torch.tensor(v, dtype=torch.float32) for k, v in model['models'][method].items()}
        _, weights = gate_scores(params, x, valid)
        if not weights.isfinite().all() or weights[~valid, 2].count_nonzero():
            raise ValueError('Invalid confidence or unmasked Face')
        fused = mix(weights, z)
        row = report[method]
        verify_metrics(fused, data['targets'], data['reliable'], row)
        np.testing.assert_allclose(group_weights(weights, data['reliable']), row['group_weights'],
                                   rtol=0, atol=1e-7)
        result['w_'+suffix] = weights.numpy()
        result['logits_'+suffix] = fused.numpy()
        result['ap_'+suffix] = np.asarray(row['metrics']['all']['per_class_ap'])
    means = torch.tensor(model['val_means']['C_tcp'], dtype=torch.float32)[valid.long()]
    mean_scores = scores(mix(means, z), data['targets'])
    np.testing.assert_allclose(mean_scores['mAP'], report['C_tcp']['group_mean_mAP'], rtol=0, atol=1e-7)
    result.update(c_mean_weights=means.numpy(), ap_c_mean=np.asarray(mean_scores['per_class_ap']),
                  metadata=dict(task=task, split=split, source_mode=source_mode, people=len(z), reliable_face=int(valid.sum()),
                                source=identity, fusion_model_sha256=sha256_file(model_path),
                                validation_lock_sha256=lock_hash, source_result_sha256=sha256_file(report_path)))
    return result
