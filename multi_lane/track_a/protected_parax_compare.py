"""Compare the completed four-arm pilot without consulting test data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

METHODS=("baseline", "frozen_pool", "fresh_only", "reuse_old")


def compare(root: Path):
    summaries={method: json.loads((root/method/'seed_summary.json').read_text()) for method in METHODS}
    audits={method: json.loads((root/method/'paired_protocol_audit.json').read_text()) for method in METHODS}
    route_audits={method: json.loads((root/method/'protected_route_audit.json').read_text()) for method in METHODS}
    expected=audits['baseline']
    for method in METHODS:
        s,a=summaries[method],audits[method]
        if s['status']!='complete' or len(s['task_metrics'])!=3 or s['test_metrics'] is not None:
            raise RuntimeError(f'{method}: incomplete pilot or unexpected test access')
        if not a['frozen_visual_unchanged'] or a['initial_selector_prompt_classifier_sha256']!=expected['initial_selector_prompt_classifier_sha256']:
            raise RuntimeError(f'{method}: unpaired initialization or changed CLIP')
        history=json.loads((root/method/'training_history.json').read_text())
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
            if not r['protected_tensors_unchanged'] or r['initial_logit_max_difference']>1e-5:
                raise RuntimeError(f'{method}: protected state or identity failed')
    # Same initialization and training at task0 in all routed arms.
    task0=[summaries[m]['task_metrics'][0]['mAP'] for m in METHODS[1:]]
    if max(task0)-min(task0)>1e-5:
        raise RuntimeError('Routed arms differ at task0 despite identical active paths')
    result={method: {'metrics':s['metrics'], 'peak_reserved_mib':audits[method]['peak_reserved_mib'],
                     'old_task_anchor_drift':route_audits[method]['tasks']['2']['old_task_anchor_drift']}
            for method,s in summaries.items()}
    result['comparisons']={
        'reuse_minus_fresh_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['fresh_only']['metrics']['final_mAP'],
        'reuse_minus_baseline_final_mAP':summaries['reuse_old']['metrics']['final_mAP']-summaries['baseline']['metrics']['final_mAP'],
        'note':'Old parameter equality is strict; numerical anchor drift is reported against the paired baseline floor. Single seed pilot is not proof of generalization.'}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    compare(parser.parse_args().root)
