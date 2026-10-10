"""Audit protected policies; explicitly locked full runs also summarize test."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

from .runner import average_precision

METHODS=("baseline", "frozen_pool", "fresh_only", "reuse_old")


def new_class_map(row, task_sizes):
    return float(np.mean(row['per_class_ap'][sum(task_sizes[:row['task_id']]):]))


def fixed_old_cohort(root, method, last_task=2, first_task=0, split="val"):
    with np.load(root/method/f'{split}_scores/task{first_task}.npz') as first, np.load(root/method/f'{split}_scores/task{last_task}.npz') as last:
        ids = last['sample_ids'].tolist()
        if len(set(ids)) != len(ids) or len(set(first['sample_ids'].tolist())) != len(first['sample_ids']):
            raise RuntimeError('Duplicate evaluation sample IDs')
        positions = {value:i for i,value in enumerate(ids)}
        indices = np.asarray([positions[value] for value in first['sample_ids']])
        n = len(first['class_indices'])
        labels = first['targets']
        if not np.array_equal(labels,last['targets'][indices,:n]) or not np.array_equal(first['class_indices'],last['class_indices'][:n]):
            raise RuntimeError('Fixed old cohort labels/classes differ')
        def score(values):
            return float(np.mean([100*average_precision(values[:,j],labels[:,j]) for j in range(n)]))
        before,after = score(first['probabilities']),score(last['probabilities'][indices,:n])
        delta = last['logits'][indices,:n]-first['logits']
        return {'samples':len(indices),'before_mAP':before,'after_mAP':after,'mAP_change':after-before,
                'saved_AMP_logit_mean_absolute_change':float(np.abs(delta).mean()),
                'saved_AMP_logit_max_absolute_change':float(np.abs(delta).max())}


def compare(root: Path, full_evaluation=False):
    count = 8 if full_evaluation else 3
    last_task = count - 1
    summaries={method: json.loads((root/method/'seed_summary.json').read_text()) for method in METHODS}
    audits={method: json.loads((root/method/'paired_protocol_audit.json').read_text()) for method in METHODS}
    route_audits={method: json.loads((root/method/'protected_route_audit.json').read_text()) for method in METHODS}
    expected=audits['baseline']
    for method in METHODS:
        s,a=summaries[method],audits[method]
        if s['config'].get('protected_parax_staged_training',False):
            raise RuntimeError('Joint-training comparison must not silently include staged fits')
        if (s['status']!='complete' or len(s['task_metrics'])!=count
                or (full_evaluation and (s['test_metrics'] is None or len(s['test_task_metrics'])!=count))
                or (not full_evaluation and s['test_metrics'] is not None)):
            raise RuntimeError(f'{method}: incomplete pilot or unexpected test access')
        if not a['frozen_visual_unchanged'] or a['initial_selector_prompt_classifier_sha256']!=expected['initial_selector_prompt_classifier_sha256']:
            raise RuntimeError(f'{method}: unpaired initialization or changed CLIP')
        if full_evaluation and (not s['config'].get('protected_parax_full_evaluation')
                                or s['config']['seed'] != summaries['baseline']['config']['seed']):
            raise RuntimeError(f'{method}: incompatible full evaluation protocol')
        history=json.loads((root/method/'training_history.json').read_text())
        if full_evaluation:
            config = s['config']
            for key in ('seed', 'training_protocol', 'epochs_per_task', 'optimizer_updates_per_task',
                        'train_batch_size', 'eval_batch_size', 'view_fusion', 'training_loss_mode', 'loss_routing'):
                if config[key] != summaries['baseline']['config'][key]:
                    raise RuntimeError(f'{method}: unpaired configuration {key}')
            if [row['task_id'] for row in s['task_metrics']] != list(range(count)) or [row['task_id'] for row in s['test_task_metrics']] != list(range(count)):
                raise RuntimeError(f'{method}: missing or duplicate task endpoints')
            budget = sum(config['optimizer_updates_per_task'] or
                         config['epochs_per_task'] * math.ceil(a['tasks'][str(t)]['train_instances']/config['train_batch_size'])
                         for t in range(count))
            if s['completed_optimizer_updates'] != budget:
                raise RuntimeError(f'{method}: incomplete optimizer budget')
            if any(not math.isfinite(v) for v in s['test_metrics'].values() if isinstance(v,(int,float))):
                raise RuntimeError(f'{method}: non-finite test metrics')

        if any(not math.isfinite(value) for value in s['metrics'].values() if isinstance(value, (int, float))):
            raise RuntimeError(f'{method}: non-finite summary metric')
        for rows in history.values():
            for row in rows:
                if method != 'baseline' and row.get('parax_grad_finite') != 1.0:
                    raise RuntimeError(f'{method}: invalid ParaX gradient')
                if any(value > .02001 for key, value in row.items() if key.endswith('_residual_ratio') and key.startswith('parax_')):
                    raise RuntimeError(f'{method}: actual residual exceeded bound')
        if any(row['skipped_optimizer_steps'] for rows in history.values() for row in rows):
            raise RuntimeError(f'{method}: AMP skipped updates')
        if s['completed_optimizer_updates']!=summaries['baseline']['completed_optimizer_updates']:
            raise RuntimeError(f'{method}: optimizer update budgets differ')
        for task in range(count):
            t=str(task)
            for key in ('first_train_batch_ids_sha256','train_sample_ids_sha256','validation_sample_ids_sha256','reporting_sample_ids_sha256'):
                if a['tasks'][t][key]!=expected['tasks'][t][key]:
                    raise RuntimeError(f'{method}: unpaired {key} task{task}')
            if bool(a['tasks'][t]['test_loaded']) != full_evaluation:
                raise RuntimeError('Unexpected test access policy')
            if full_evaluation and a['tasks'][t]['test_sample_ids_sha256'] != expected['tasks'][t]['test_sample_ids_sha256']:
                raise RuntimeError(f'{method}: unpaired test samples task{task}')
            for split in (('val', 'test') if full_evaluation else ('val',)):
                with np.load(root/'baseline'/f'{split}_scores/task{task}.npz') as base_scores, np.load(root/method/f'{split}_scores/task{task}.npz') as scores:
                    for key in ('sample_ids', 'class_indices', 'targets'):
                        if not np.array_equal(base_scores[key], scores[key]):
                            raise RuntimeError(f'{method}: unpaired {split} {key} task{task}')
                    if not np.isfinite(scores['probabilities']).all():
                        raise RuntimeError(f'{method}: non-finite {split} probabilities')
            r=route_audits[method]['tasks'][t]
            identity = r.get('identity_audit', {})
            if not r['protected_tensors_unchanged'] or not identity.get('passed', False):
                raise RuntimeError(f'{method}: protected state or identity failed')
            if identity.get('schema_version') != 2:
                raise RuntimeError(f'{method}: missing separated identity audit')
    # Same initialization and training at task0 in all routed arms.
    task0=[summaries[m]['task_metrics'][0]['mAP'] for m in METHODS[1:]]
    if max(task0)-min(task0)>1e-5:
        raise RuntimeError('Routed arms differ at task0 despite identical active paths')
    result={method: {'metrics':s['metrics'], 'peak_reserved_mib':audits[method]['peak_reserved_mib'],
                     'residual_disabled_metrics':s.get('residual_disabled_metrics'),
                     'residual_disabled_task_metrics':s.get('residual_disabled_task_metrics'),
                     'fixed_task0_cohort':fixed_old_cohort(root,method,last_task),
                     'per_task_old_new_mAP':[
                         {'task_id':row['task_id'],
                          'old_classes_mAP':float(np.mean(row['per_class_ap'][:sum(s['config']['task_sizes'][:row['task_id']])])) if row['task_id'] else None,
                          'new_classes_mAP':new_class_map(row,s['config']['task_sizes'])}
                         for row in s['task_metrics']],
                     'old_task_anchor_drift':route_audits[method]['tasks'][str(last_task)]['old_task_anchor_drift']}
            for method,s in summaries.items()}
    result['comparisons']={
        'reuse_minus_fresh_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['fresh_only']['metrics']['final_mAP'],
        'reuse_minus_baseline_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['baseline']['metrics']['final_mAP'],
        'note':'Old parameter equality is strict; numerical anchor drift is reported against the paired baseline floor. Single seed pilot is not proof of generalization.'}
    for method in METHODS[1:]:
        disabled=summaries[method].get('residual_disabled_task_metrics',[])
        if len(disabled)!=count:
            raise RuntimeError(f'{method}: missing residual-disabled inference')
        result[method]['residual_ablation']=[{
            'task_id':row['task_id'],
            'enabled_minus_disabled_mAP':row['mAP']-off['mAP'],
            'disabled_minus_independent_baseline_mAP':off['mAP']-base['mAP'],
            'new_class_enabled_minus_disabled_mAP':new_class_map(row,summaries[method]['config']['task_sizes'])-new_class_map(off,summaries[method]['config']['task_sizes']),
            'new_class_disabled_minus_baseline_mAP':new_class_map(off,summaries[method]['config']['task_sizes'])-new_class_map(base,summaries['baseline']['config']['task_sizes']),
        } for row,off,base in zip(summaries[method]['task_metrics'],disabled,summaries['baseline']['task_metrics'])]
    if full_evaluation:
        for method, summary in summaries.items():
            result[method]['test_metrics'] = summary['test_metrics']
            result[method]['test_task_metrics'] = summary['test_task_metrics']
            result[method]['fixed_old_cohorts'] = {
                split: {str(t): fixed_old_cohort(root, method, last_task, t, split)
                        for t in range(last_task)} for split in ('val', 'test')}
            if method != 'baseline':
                disabled = summary.get('residual_disabled_test_task_metrics', [])
                if len(disabled) != count:
                    raise RuntimeError(f'{method}: missing residual-disabled test inference')
                result[method]['residual_disabled_test_metrics'] = summary['residual_disabled_test_metrics']
                result[method]['test_residual_ablation'] = [{
                    'task_id': row['task_id'],
                    'enabled_minus_disabled_mAP': row['mAP'] - off['mAP'],
                    'disabled_minus_independent_baseline_mAP': off['mAP'] - base['mAP'],
                    'new_class_enabled_minus_disabled_mAP': new_class_map(row, summary['config']['task_sizes']) - new_class_map(off, summary['config']['task_sizes']),
                } for row, off, base in zip(summary['test_task_metrics'], disabled, summaries['baseline']['test_task_metrics'])]
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def compare_seeds(root: Path, seeds):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Need unique nonempty seeds")
    for seed in seeds:
        source=json.loads((root/f'seed{seed}'/'baseline/seed_summary.json').read_text())
        if source['seed'] != seed:
            raise RuntimeError('Seed directory does not match run seed')
    results = {str(seed): compare(root/f'seed{seed}', full_evaluation=True) for seed in seeds}
    aggregate = {'seeds': list(seeds), 'test_policy': 'fixed configurations; no selection', 'methods': {}}
    rows = []
    for method in METHODS:
        aggregate['methods'][method] = {}
        for split, key in [('validation', 'metrics'), ('test', 'test_metrics')]:
            values = {metric: [results[str(seed)][method][key][metric] for seed in seeds]
                      for metric in ('final_mAP', 'average_mAP', 'forgetting')}
            aggregate['methods'][method][split] = {
                metric: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v)>1 else 0., 'values': v}
                for metric, v in values.items()}
            for seed in seeds:
                metrics = results[str(seed)][method][key]
                rows.append({'seed': seed, 'method': method, 'split': split,
                             **{metric: metrics[metric] for metric in values}})
    (root/'multiseed_comparison.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    with (root/'multiseed_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return aggregate


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--full-evaluation', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    if args.seeds is not None:
        if not args.full_evaluation:
            parser.error('--seeds requires --full-evaluation')
        compare_seeds(args.root, args.seeds)
    else:
        compare(args.root, args.full_evaluation)
