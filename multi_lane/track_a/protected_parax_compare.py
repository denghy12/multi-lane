"""Compare the completed four-arm pilot without consulting test data."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

from .runner import average_precision

METHODS=("baseline", "frozen_pool", "fresh_only", "reuse_old")


def new_class_map(row, task_sizes):
    return float(np.mean(row['per_class_ap'][sum(task_sizes[:row['task_id']]):]))


def fixed_old_cohort(root, method, last_task=2):
    with np.load(root/method/'val_scores/task0.npz') as first, np.load(root/method/f'val_scores/task{last_task}.npz') as last:
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


def compare(root: Path):
    summaries={method: json.loads((root/method/'seed_summary.json').read_text()) for method in METHODS}
    audits={method: json.loads((root/method/'paired_protocol_audit.json').read_text()) for method in METHODS}
    route_audits={method: json.loads((root/method/'protected_route_audit.json').read_text()) for method in METHODS}
    expected=audits['baseline']
    for method in METHODS:
        s,a=summaries[method],audits[method]
        if s['config'].get('protected_parax_staged_training',False):
            raise RuntimeError('Joint-training comparison must not silently include staged fits')
        if s['status']!='complete' or len(s['task_metrics'])!=3 or s['test_metrics'] is not None:
            raise RuntimeError(f'{method}: incomplete pilot or unexpected test access')
        if not a['frozen_visual_unchanged'] or a['initial_selector_prompt_classifier_sha256']!=expected['initial_selector_prompt_classifier_sha256']:
            raise RuntimeError(f'{method}: unpaired initialization or changed CLIP')
        history=json.loads((root/method/'training_history.json').read_text())
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
        for task in range(3):
            t=str(task)
            for key in ('first_train_batch_ids_sha256','train_sample_ids_sha256','validation_sample_ids_sha256','reporting_sample_ids_sha256'):
                if a['tasks'][t][key]!=expected['tasks'][t][key]:
                    raise RuntimeError(f'{method}: unpaired {key} task{task}')
            if a['tasks'][t]['test_loaded']:
                raise RuntimeError('Test data was loaded')
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
                     'fixed_task0_cohort':fixed_old_cohort(root,method),
                     'per_task_old_new_mAP':[
                         {'task_id':row['task_id'],
                          'old_classes_mAP':float(np.mean(row['per_class_ap'][:sum(s['config']['task_sizes'][:row['task_id']])])) if row['task_id'] else None,
                          'new_classes_mAP':new_class_map(row,s['config']['task_sizes'])}
                         for row in s['task_metrics']],
                     'old_task_anchor_drift':route_audits[method]['tasks']['2']['old_task_anchor_drift']}
            for method,s in summaries.items()}
    result['comparisons']={
        'reuse_minus_fresh_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['fresh_only']['metrics']['final_mAP'],
        'reuse_minus_baseline_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['baseline']['metrics']['final_mAP'],
        'note':'Old parameter equality is strict; numerical anchor drift is reported against the paired baseline floor. Single seed pilot is not proof of generalization.'}
    for method in METHODS[1:]:
        disabled=summaries[method].get('residual_disabled_task_metrics',[])
        if len(disabled)!=3:
            raise RuntimeError(f'{method}: missing residual-disabled inference')
        result[method]['residual_ablation']=[{
            'task_id':row['task_id'],
            'enabled_minus_disabled_mAP':row['mAP']-off['mAP'],
            'disabled_minus_independent_baseline_mAP':off['mAP']-base['mAP'],
            'new_class_enabled_minus_disabled_mAP':new_class_map(row,summaries[method]['config']['task_sizes'])-new_class_map(off,summaries[method]['config']['task_sizes']),
            'new_class_disabled_minus_baseline_mAP':new_class_map(off,summaries[method]['config']['task_sizes'])-new_class_map(base,summaries['baseline']['config']['task_sizes']),
        } for row,off,base in zip(summaries[method]['task_metrics'],disabled,summaries['baseline']['task_metrics'])]
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    compare(parser.parse_args().root)
