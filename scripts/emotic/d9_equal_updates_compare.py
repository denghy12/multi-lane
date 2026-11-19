"""Compare matched-update 90% training to native 90% and full-TRAIN trajectories."""
import argparse
import math
from pathlib import Path

import numpy as np

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import VIEWS
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS

TRAJECTORIES = ('holdout', 'equal_updates', 'full')
BRANCHES = ('full', 'person', 'face', 'fixed')


def changes(holdout, equal_updates, full):
    """Fraction of a positive pre-existing deficit recovered; all differences in points."""
    gap = full - holdout
    gain = equal_updates - holdout
    return dict(gain_from_updates=gain, full_minus_matched=full - equal_updates,
                original_gap=gap, recovered_fraction=gain / gap if gap > 0 else None)


def audit_training(roots):
    configs = {k: read(p / 'config.json') for k, p in roots.items()}
    hold, matched, full = [configs[k] for k in TRAJECTORIES]
    if (matched.get('equal_updates') is None or hold['mode'] != 'sample_holdout'
            or full['mode'] != 'sample_full' or not hold['protect_branches'] or hold['smoke']
            or (hold['epochs'], hold['tasks'], hold['seed'], hold['batch_size']) != (30, 8, 1, 64)):
        raise ValueError('Unexpected three-trajectory setup')
    for key in hold:
        if matched[key] != hold[key]:
            raise ValueError(f'Matched branch/gate configuration differs from 90% at {key}')
        if key not in ('mode', 'image_split', 'fusion_asl') and full[key] != hold[key]:
            raise ValueError(f'Full and original 90% recipe differ at {key}')
    budget = matched['equal_updates']
    for trajectory, key in [('full','full_config_sha256'), ('holdout','holdout_config_sha256')]:
        require_digest(roots[trajectory]/'config.json', budget[key])
    roles_hash = budget['roles_sha256']
    require_digest(roots['holdout']/'train_image_roles.json', roles_hash)
    require_digest(roots['equal_updates'] / 'train_image_roles.json', roles_hash)
    for p in roots.values():
        status = read(p/'training_complete.json')
        if status['status'] != 'complete' or status['tasks'] != len(TASK_SIZES) or status['smoke']:
            raise ValueError(f'Incomplete formal training at {p}')
    tasks = []
    for t in range(len(TASK_SIZES)):
        cp = Path('cohorts') / f'task{t}.json'
        hist = Path('history') / f'task{t}.json'
        target = budget['tasks'][str(t)]
        require_digest(roots['full']/hist, target['full_history_sha256'])
        require_digest(roots['full']/cp, target['full_cohort_sha256'])
        require_digest(roots['holdout']/cp, target['holdout_cohort_sha256'])
        cohorts = {k: read(p/cp) for k,p in roots.items()}
        history = {k: read(p/hist) for k,p in roots.items()}
        if (cohorts['holdout'] != cohorts['equal_updates']
                or history['holdout']['coverage'] != history['equal_updates']['coverage']):
            raise ValueError(f'90% branch or fusion people differ at task{t}')
        rows = {k: record['epochs'] for k,record in history.items()}
        if any(len(v) != hold['epochs'] for v in rows.values()):
            raise ValueError(f'Incomplete task{t} training history')
        first_rates = np.asarray(rows['equal_updates'][0]['branch_lrs'], dtype=float)
        if set(first_rates.tolist()) != {.0125, .0004}:
            raise ValueError(f'Unexpected initial branch learning rates at task{t}')
        people = cohorts['holdout']['branch']['people']
        extra_per_person = np.zeros(people, dtype=np.int64)
        for epoch,(original,new,reference) in enumerate(zip(*(rows[k] for k in TRAJECTORIES)), start=1):
            if any(r['skipped'] or (r['task'],r['epoch']) != (t,epoch) for r in (original,new,reference)):
                raise ValueError(f'Skipped or unordered update, task{t} epoch{epoch}')
            if (new['updates'] != reference['updates'] or new['examples'] != reference['examples']
                    or original['gate_schedule'] != new['gate_schedule']
                    or any(r['examples'] != cohorts[k]['branch']['people'] for k,r in
                           [('holdout',original), ('full',reference)])):
                raise ValueError(f'Exposure, update or fusion-sweep mismatch, task{t} epoch{epoch}')
            sampling = new['sampling']
            repeats = np.asarray(sampling['extra_indices'], dtype=np.int64)
            if (sampling['unique_people'] != people or sampling['examples'] != new['examples']
                    or len(repeats) != sampling['extra_examples'] or len(repeats) != new['examples'] - people
                    or sampling['duplicate_within_batches'] or np.any(repeats < 0) or np.any(repeats >= people)
                    or sampling['batch_sizes'] != [hold['batch_size']] * (new['updates'] - 1) + [target['last_batch_size']]):
                raise ValueError(f'90% exposure mismatch, task{t} epoch{epoch}')
            extra_per_person += np.bincount(repeats, minlength=people)
            if extra_per_person.max() - extra_per_person.min() > 1:
                raise ValueError(f'Unbalanced 90% repetitions at task{t} epoch{epoch}')
            factor = (1 + math.cos(math.pi * (epoch - 1) / hold['epochs'])) / 2
            np.testing.assert_allclose(new['branch_lrs'], first_rates*factor, rtol=1e-12, atol=1e-16)
        tasks.append(dict(task=t,
                          branch_people={k:c['branch']['people'] for k,c in cohorts.items()},
                          fusion_people={k:c['fusion']['people'] for k,c in cohorts.items()},
                          updates={k:sum(r['updates'] for r in v) for k,v in rows.items()},
                          examples={k:sum(r['examples'] for r in v) for k,v in rows.items()},
                          supplemental_exposures_per_person_range=[int(extra_per_person.min()),int(extra_per_person.max())],
                          fp32_retries={k:sum(r['fp32_retries'] for r in v) for k,v in rows.items()},
                          history_sha256={k:sha256_file(p/hist) for k,p in roots.items()},
                          cohort_sha256={k:sha256_file(p/cp) for k,p in roots.items()}))
    return dict(holdout_roles_sha256=roles_hash, tasks=tasks,
                total_updates={k:sum(v['updates'][k] for v in tasks) for k in roots})


def read_population(root, split, t, row, cfg_hash, validation_hash):
    path = root/'evaluation'/split/f'task{t}.npz'
    protocol = row['protocol']
    expected = dict(split=split, smoke=False, task=t, config_sha256=cfg_hash, batch_size=64,
                    validation_result_sha256=validation_hash)
    if any(protocol[k] != v for k,v in expected.items()):
        raise ValueError(f'Official archive protocol changed: {path}')
    require_digest(path, row['scores_sha256'])
    require_digest(root/'compact_checkpoints'/f'task{t}.pth', protocol['checkpoint_sha256'])
    with np.load(path, allow_pickle=False) as a:
        if tuple(a['task_sizes']) != tuple(TASK_SIZES[:t+1]) or tuple(a['view_order']) != VIEWS:
            raise ValueError(f'Changed task/view order: {path}')
        return {k:a[k] for k in ('sample_ids','targets','reliable')}


def compare(root, historical, out, split):
    roots = dict(holdout=historical/'sample_holdout', equal_updates=root, full=historical/'sample_full')
    out.mkdir(parents=True, exist_ok=True)
    seal_path = out/'validation_seal.json'
    if split == 'test':
        seal = read(seal_path)
        require_digest(out/'val_comparison.json', seal['comparison_sha256'])
        val_hashes = seal['official_validation_sha256']
        for k,p in roots.items():
            require_digest(p/'evaluation/val/result.json', val_hashes[k])
    else:
        val_hashes = dict.fromkeys(roots)
    audit = audit_training(roots)
    paths = {k:p/'evaluation'/split/'result.json' for k,p in roots.items()}
    reports = {k:read(p) for k,p in paths.items()}
    hashes = {k:sha256_file(p/'config.json') for k,p in roots.items()}
    for k,r in reports.items():
        if (r['split'] != split or r['smoke'] or len(r['stages']) != len(TASK_SIZES)
                or r['mode'] != ('sample_full' if k=='full' else 'sample_holdout')):
            raise ValueError(f'Unexpected official evaluation {paths[k]}')
    stages = []
    summaries = {}
    for t in range(len(TASK_SIZES)):
        data = {k:read_population(p,split,t,reports[k]['stages'][t],hashes[k],val_hashes[k]) for k,p in roots.items()}
        for key,v in data['holdout'].items():
            for k in ('equal_updates','full'):
                np.testing.assert_array_equal(v,data[k][key],err_msg=f'{split}/task{t}/{k}/{key}')
        stages.append(dict(task=t, people=len(data['holdout']['reliable']),
                           reliable_face=int(data['holdout']['reliable'].sum()),
                           scores_sha256={k:r['stages'][t]['scores_sha256'] for k,r in reports.items()},
                           mAP={m:{g:{k:r['stages'][t]['metrics'][m][g]['mAP'] for k,r in reports.items()}
                                   for g in GROUPS} for m in BRANCHES}))
    for method in BRANCHES:
        summaries[method] = {}
        for group in GROUPS:
            by_traj = {}
            for k,r in reports.items():
                row = r['summary'][method][group]
                verified = summarize_tasks([TaskMetrics(**s['metrics'][method][group]) for s in r['stages']])
                if row != verified:
                    raise ValueError(f'Official summary mismatch: {split}/{k}/{method}/{group}')
                by_traj[k] = {key:row[key] for key in MAP_KEYS}
            compare_keys = {key:changes(*(by_traj[k][key] for k in TRAJECTORIES)) for key in MAP_KEYS}
            summaries[method][group] = dict(values=by_traj, difference=compare_keys)
    gate_refs = reports['equal_updates']['comparisons']
    new_hashes = {k:sha256_file(p) for k,p in paths.items()}
    report = dict(split=split, source_config_sha256=hashes, official_result_sha256=new_hashes,
                  training=audit, stages=stages, branch_summary=summaries,
                  simultaneous_gate_readouts={
                      name:dict(scores={key:reports['equal_updates']['summary'][name]['all'][key] for key in MAP_KEYS},
                                comparisons=gate_refs[name])
                      for name in reports['equal_updates']['learned_readouts']})
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(seal_path)
    dest = out/f'{split}_comparison.json'
    write_verified(dest,report)
    if split=='val':
        write_verified(seal_path,dict(comparison_sha256=sha256_file(dest),official_validation_sha256=new_hashes))
    for m in ('full','fixed'):
        print(split,m,summaries[m]['all'], flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('root','historical','out'):
        parser.add_argument('--'+option,required=True,type=Path)
    parser.add_argument('--split',choices=('val','test'),required=True)
    compare(**vars(parser.parse_args()))
