"""Evaluate held-TRAIN paired A on matched-step Full/Person and original-budget Face."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, paired, predictions, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, PROTECTED_BASELINE
from scripts.emotic.fit_d9_task10_probe import weights

# Pre-existing fixed Face-only archival swap, reported before fitting this candidate.
SWAP_FIXED_REFERENCE = {
    'val': (43.53742362011429, 51.16335550957131),
    'test': (33.058273885960865, 40.37966760631784),
}


def validation_decision(candidate, fixed, current):
    return (all(candidate[key] >= fixed[key] for key in MAP_KEYS)
            and any(candidate[key] >= current[key] for key in MAP_KEYS))


def stage_readouts(new, old, centers, current_centers, stage):
    for key in ('sample_ids', 'targets', 'reliable', 'task_sizes', 'view_order', 'fixed_weights'):
        np.testing.assert_array_equal(new[key], old[key], err_msg=f'Stage {stage} alignment: {key}')
    current, target, reliable = predictions(new, current_centers, stage)
    # Verify the original-budget archive's schema and native controls as well.
    original, _, _ = predictions(old, current_centers, stage)
    mixed = torch.from_numpy(new['branch_logits'].copy())
    mixed[..., 2] = torch.from_numpy(old['branch_logits'][..., 2])
    sizes = TASK_SIZES[:stage + 1]
    by_task = [weights(torch.tensor(centers[str(t)]['center'], dtype=torch.float32), reliable)
               for t in range(stage + 1)]
    candidate_weights = torch.stack(by_task, dim=1)
    if not torch.isfinite(candidate_weights).all() or candidate_weights[~reliable, :, 2].count_nonzero():
        raise ValueError('Invalid held-TRAIN weights or unreliable Face contribution')
    scores = {
        'A': weighted_logits(mixed, candidate_weights, sizes),
        'fixed': weighted_logits(mixed, torch.from_numpy(new['fixed_weights']), sizes),
        'current_A': current['A'],
        'current_fixed': current['fixed'],
        'old_fixed': original['fixed'],
    }
    torch.testing.assert_close(scores['fixed'][~reliable], scores['current_fixed'][~reliable], atol=2e-5, rtol=2e-6)
    return scores, target, reliable


def run(matched, old, out, split):
    lock_path = out / 'centers_lock.json'
    lock_hash = sha256_file(lock_path)
    lock = read(lock_path)
    seal_path = out / 'validation_seal.json'
    roots = dict(matched=matched, old=old)
    seal = None
    if split == 'test':
        seal = read(seal_path)
        require_digest(lock_path, seal['centers_sha256'])
        require_digest(out / 'val_results.json', seal['val_results_sha256'])
        for name, root in roots.items():
            require_digest(root/'evaluation/val/result.json', seal['native_val_sha256'][name])
    native, native_sha = {}, {}
    for name, root in roots.items():
        require_digest(root/'config.json', lock[name + '_config_sha256'])
        path = root/'evaluation'/split/'result.json'
        native[name], native_sha[name] = read(path), sha256_file(path)
        if (native[name]['smoke'] or native[name]['split'] != split or native[name]['mode'] != 'sample_holdout'
                or len(native[name]['stages']) != len(TASK_SIZES)):
            raise ValueError(f'Expected complete {split} evaluation of {name}')
    current_dir = matched / 'paired_A'
    current_result = read(current_dir/f'{split}_results.json')
    current_lock_path = current_dir/'centers_lock.json'
    require_digest(current_lock_path, current_result['centers_sha256'])
    if current_result['native_result_sha256'] != native_sha['matched']:
        raise ValueError('Current paired A was evaluated on different predictions')
    current_lock = read(current_lock_path)
    current_centers, rows = {}, []
    for stage in range(len(TASK_SIZES)):
        t = str(stage)
        center_path = current_dir / f'task{stage}_center.json'
        require_digest(center_path, current_lock[t])
        current_centers[t] = read(center_path)
        candidate = lock['tasks'][t]
        digests = {}
        for name, root in roots.items():
            row = native[name]['stages'][stage]
            expected = dict(task=stage, split=split, smoke=False,
                            config_sha256=lock[name + '_config_sha256'],
                            checkpoint_sha256=candidate[name + '_checkpoint_sha256'],
                            validation_result_sha256=seal['native_val_sha256'][name] if seal else None)
            if any(row['protocol'][k] != v for k, v in expected.items()):
                raise ValueError(f'Fit and evaluation sources differ: {name} task{stage}')
            archive = root/'evaluation'/split/f'task{stage}.npz'
            digests[name] = row['scores_sha256']
            require_digest(archive, digests[name])
        torch.set_num_threads(1)
        archive_name = Path('evaluation')/split/f'task{stage}.npz'
        with np.load(matched/archive_name, allow_pickle=False) as new, np.load(old/archive_name, allow_pickle=False) as src:
            scores, y, valid = stage_readouts(new, src, lock['tasks'], current_centers, stage)
        torch.set_num_threads(2)
        metrics = metrics_for_stage(stage, y, valid, scores)
        expected_replays = dict(current_A=current_result['stages'][stage]['metrics']['A'],
                                current_fixed=native['matched']['stages'][stage]['metrics']['fixed'],
                                old_fixed=native['old']['stages'][stage]['metrics']['fixed'])
        for name, historical in expected_replays.items():
            if metrics[name] != historical:
                raise ValueError(f'Native score replay failed for {name}, {split}, stage {stage}')
        rows.append(dict(stage=stage, people=len(y), reliable_face=int(valid.sum()),
                         archive_sha256=digests, metrics=metrics, paired=paired(metrics, stage)))
    summary = {name: {g: summarize_tasks([TaskMetrics(**row['metrics'][name][g]) for row in rows])
                      for g in GROUPS} for name in rows[0]['metrics']}
    overall = {name: group['all'] for name, group in summary.items()}
    np.testing.assert_allclose([overall['fixed'][k] for k in MAP_KEYS], SWAP_FIXED_REFERENCE[split],
                               atol=1e-9, rtol=0, err_msg='Prior fixed Face-budget swap failed to replay')
    references = dict(same_branches_fixed=overall['fixed'], current_A=overall['current_A'],
                      full_train_fixed=PROTECTED_BASELINE[split])
    delta = {name: {k: overall['A'][k] - ref[k] for k in MAP_KEYS} for name, ref in references.items()}
    report = dict(split=split, stages=rows, summary=summary, A_minus_reference=delta,
                  protected_full_train_fixed=PROTECTED_BASELINE[split],
                  centers_sha256=lock_hash, source_result_sha256=native_sha,
                  current_paired_A_result_sha256=sha256_file(current_dir/f'{split}_results.json'))
    if split == 'val':
        report['advance_formal'] = validation_decision(overall['A'], overall['fixed'], overall['current_A'])
    else:
        report['validation_seal_sha256'] = sha256_file(seal_path)
        report['advance_formal_from_val'] = read(out/'val_results.json')['advance_formal']
    dest = out/f'{split}_results.json'
    write_verified(dest, report)
    if split == 'val':
        write_verified(seal_path, dict(centers_sha256=lock_hash, val_results_sha256=sha256_file(dest),
                                      native_val_sha256=native_sha))
    print(split, {name: {k: overall[name][k] for k in MAP_KEYS} for name in overall}, delta, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matched', 'old', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    run(**vars(parser.parse_args()))
