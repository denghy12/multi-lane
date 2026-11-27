"""Evaluate the unchanged four summary-logit readouts over all eight stages, twice."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from multi_lane.track_a.summary_logit_fusion import METHODS, NAMES, SummaryLogitGate, fused_logits, masked_weights
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import extract_train, native_scores
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_extract import extract
from scripts.emotic.d9_summary_readout import (
    load_stage, train, predictions, save_npz, coverage, group_mean_scores, permutation,
)
from scripts.emotic.d9_summary_run import config_check as pilot_check, protocol, ROOT
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, write_json


CONFIG = ROOT/'scripts/emotic/configs/d9_summary_eight_20261127.json'
SCOPE = 'current_stage_gate_broadcast_all_seen'
FIT_PROTOCOL = 'stage-local TRAIN BCE from original initialization; one current-stage gate for every seen head'
# User-requested original D benchmark, NOT the later protected-source fixed scores.
BENCHMARK = dict(val=dict(final_mAP=44.0329, average_mAP=51.1335),
                 test=dict(final_mAP=33.1770, average_mAP=40.2083))
KEYS = ('final_mAP', 'average_mAP')


def config_check(cfg):
    if cfg['tasks'] != list(range(8)) or cfg['evaluation'] != SCOPE:
        raise ValueError('Exactly eight chronological stages and one gate for all seen heads are required')
    copy = deepcopy(cfg)
    copy['tasks'] = [0, 3, 6]
    pilot_check(copy)
    if cfg['diagnostics'] != 'three_train_folds_and_100_permutations; no_bootstrap_in_main_evaluation':
        raise ValueError('Unexpected diagnostic scope')


def aggregate(rows, smoke):
    tasks = [r['task'] for r in rows]
    if not smoke and tasks != list(range(8)):
        raise ValueError('Partial/current-head results cannot be reported as an eight-stage result')
    for row in rows:
        for method in METHODS:
            m = row['metrics'][method]['all']
            if m['task_id'] != row['task'] or m['seen_classes'] != sum(TASK_SIZES[:row['task']+1]):
                raise ValueError('Expected all seen classes in every stage')
    if smoke:
        return None
    return {method: {group: summarize_tasks([TaskMetrics(**row['metrics'][method][group]) for row in rows])
                     for group in ('all', 'reliable', 'unreliable')
                     if all(group in row['metrics'][method] for row in rows)} for method in METHODS}


def canonical_fixed(source, features, split, task, stage):
    path, _, _, _ = native_scores(source, features, split, task)
    with np.load(path, allow_pickle=False) as z:
        positions = {str(key): i for i, key in enumerate(z['sample_ids'])}
        take = np.asarray([positions[str(key)] for key in stage.person_ids])
        np.testing.assert_array_equal(z['targets'][take], stage.targets.numpy())
        np.testing.assert_array_equal(z['reliable'][take], stage.reliable.numpy())
        return torch.from_numpy(z['fixed_logits'][take]).float()


def evaluate(source, features, out, split, tasks, config, smoke):
    # Match the historical eight-stage evaluator's sigmoid reduction/thread setting.
    torch.set_num_threads(2)
    fitted = read(out/'fit.json')
    if (fitted['tasks'] != tasks or fitted['config'] != config or fitted['smoke'] != smoke
            or fitted['protocol'] != FIT_PROTOCOL):
        raise ValueError('Unexpected fitting protocol')
    seal = read(out/'validation_seal.json') if split == 'test' else None
    if seal:
        require_digest(out/'fit.json', seal['fit_sha256'])
        require_digest(out/'val_results.json', seal['val_sha256'])
    native = read(source/'evaluation'/split/'result.json')
    if native['smoke'] or native['split'] != split or len(native['stages']) != 8:
        raise ValueError('Incomplete native eight-stage source')
    rows, mean_locks = [], {}
    for state in fitted['stages']:
        task = state['task']
        stage = load_stage(source, features, split, task, smoke, current_only=False)
        for key in ('checkpoint_sha256', 'config_sha256'):
            if state['provenance'][key] != stage.provenance[key]:
                raise ValueError('Fitting and evaluation branch identities differ')
        actual, arrays, controls, locks = {}, {}, {}, {}
        for method in METHODS:
            gate = SummaryLogitGate.restore(state['methods'][method]['gate'])
            scores, logits = predictions(stage, gate)
            if method == 'fixed':
                exact = canonical_fixed(source, features, split, task, stage)
                torch.testing.assert_close(logits, exact, rtol=2e-6, atol=5e-6)
                # Preserve native float arithmetic and ties for the paired fixed control.
                logits = exact
            actual[method] = logits
            arrays[method+'_scores'] = scores.numpy()
            arrays[method+'_weights'] = masked_weights(scores[stage.inverse], stage.reliable).numpy()
            arrays[method+'_logits'] = logits.numpy()
            if method in METHODS[2:]:
                locked = None if seal is None else seal['mean_scores'][str(task)][method]
                mean_scores, locks[method] = group_mean_scores(scores, stage, locked)
                mean_logits = fused_logits(stage.logits, mean_scores, stage.inverse, stage.reliable)
                mean = metrics_for_stage(task, stage.targets, stage.reliable, {method: mean_logits})[method]
                rng, repeats = np.random.default_rng(config['permutation_seed']), []
                for _ in range(config['permutations']):
                    shuffled = fused_logits(stage.logits, scores[permutation(stage, rng)], stage.inverse, stage.reliable)
                    repeats.append(metrics_for_stage(task, stage.targets, stage.reliable,
                                                      {method: shuffled})[method]['all']['mAP'])
                controls[method] = dict(mean=mean, permutation_mAP=repeats,
                                       permutation_mean=float(np.mean(repeats)))
                arrays[method+'_mean_logits'] = mean_logits.numpy()
        metrics = metrics_for_stage(task, stage.targets, stage.reliable, actual)
        if not smoke:
            archived = native['stages'][task]
            if stage.provenance['native_sha256'] != archived['scores_sha256']:
                raise ValueError('Native stage summary and scores disagree')
            for group, observed in metrics['fixed'].items():
                for key, value in observed.items():
                    np.testing.assert_allclose(value, archived['metrics']['fixed'][group][key], rtol=0, atol=1e-9,
                                               err_msg=f'Fixed native metric replay: {split}/{task}/{group}/{key}')
        path = out/split/f'task{task}.npz'
        save_npz(path, image_ids=stage.ids, person_ids=stage.person_ids, inverse=stage.inverse.numpy(),
                 targets=stage.targets.numpy(), reliable=stage.reliable.numpy(), counts=stage.counts,
                 reliable_counts=stage.reliable_counts, **arrays)
        current = TASK_SIZES[task]
        deltas = {}
        for method in METHODS:
            diff = np.asarray(metrics[method]['all']['per_class_ap']) - np.asarray(metrics['fixed']['all']['per_class_ap'])
            deltas[method] = dict(all=float(diff.mean()), current=float(diff[-current:].mean()),
                                 old=float(diff[:-current].mean()) if len(diff) > current else None)
        rows.append(dict(task=task, coverage=coverage(stage), provenance=stage.provenance,
                         metrics=metrics, controls=controls, delta_vs_fixed=deltas,
                         predictions_sha256=sha256_file(path)))
        mean_locks[str(task)] = locks
        print(f'{split} stage{task} all {stage.targets.shape[1]} seen classes: '
              +json.dumps({m: metrics[m]['all']['mAP'] for m in METHODS}), flush=True)
    summary = aggregate(rows, smoke)
    comparisons = None
    if summary is not None:
        for key in KEYS:
            np.testing.assert_allclose(summary['fixed']['all'][key], native['summary']['fixed']['all'][key],
                                       rtol=0, atol=1e-9)
        comparisons = {m: {label: dict(delta={k: summary[m]['all'][k]-ref[k] for k in KEYS},
                              both_at_least_reference=all(summary[m]['all'][k] >= ref[k] for k in KEYS))
                          for label, ref in dict(original_D=BENCHMARK[split],
                                                same_branches_fixed=summary['fixed']['all']).items()} for m in METHODS}
    result = dict(split=split, smoke=smoke, evaluation=SCOPE, fit_sha256=sha256_file(out/'fit.json'),
                  native_result_sha256=sha256_file(source/'evaluation'/split/'result.json'),
                  stages=rows, summary=summary, comparisons=comparisons, original_D=BENCHMARK[split],
                  target_comparable=not smoke)
    write_verified(out/f'{split}_results.json', result)
    if split == 'val':
        write_verified(out/'validation_seal.json', dict(fit_sha256=sha256_file(out/'fit.json'),
                       val_sha256=sha256_file(out/'val_results.json'), mean_scores=mean_locks))
    return result


def reuse_artifacts(cfg, base):
    """Only immutable experiment data are copied; all metadata are revalidated by loaders."""
    previous = Path(cfg['reuse'])
    prior = read(previous/'protocol.json')
    if prior['sources'] != {k: {**v, 'checkpoints': {t: h for t, h in v['checkpoints'].items() if int(t) in (0, 3, 6)}}
                             for k, v in protocol(cfg)['sources'].items()}:
        raise ValueError('Prior pilot used different source checkpoints')
    require_digest(previous/'comparison.json', read(previous/'complete.json')['comparison_sha256'])
    for coverage_name in cfg['sources']:
        for directory in ('features', 'native_train'):
            for path in (previous/coverage_name/directory).rglob('*'):
                if path.is_file() and path.suffix in ('.json', '.npz'):
                    dest = base/coverage_name/directory/path.relative_to(previous/coverage_name/directory)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        require_digest(dest, sha256_file(path))
                    else:
                        shutil.copyfile(path, dest)


def run(cfg, phase):
    config_check(cfg)
    record = protocol(cfg)
    if phase == 'check':
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Run only committed implementation/configuration')
    smoke = phase == 'preflight'
    base = Path(cfg['output'])/phase
    base.mkdir(parents=True, exist_ok=True)
    if not smoke:
        prior = read(Path(cfg['output'])/'preflight'/'complete.json')
        if prior['digest'] != record['digest'] or prior['sources'] != record['sources']:
            raise ValueError('This exact protocol has not passed preflight')
    write_verified(base/'protocol.json', record)
    tasks = [0, 7] if smoke else cfg['tasks']
    fitting = dict(cfg['fit'])
    if smoke:
        fitting.update(steps=5, permutations=2)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                        '-p', 'test_d9_summary*.py', '-v'], cwd=ROOT, check=True)
    else:
        reuse_artifacts(cfg, base)
    for name, info in cfg['sources'].items():
        source, out = Path(info['path']), base/name
        for task in tasks:
            if info['mode'] == 'sample_full':
                extract_train(source, out/'native_train', task, **cfg['extraction'])
            extract(source, out/'features', 'train', task, smoke=smoke, **cfg['extraction'])
        train(source, out/'features', out/'readout', tasks, fitting, smoke, protocol=FIT_PROTOCOL)
    for split in ('val', 'test'):
        # Both sources complete and seal all Validation stages before either Test evaluation.
        if split == 'test':
            for name in cfg['sources']:
                directory = base/name/'readout'
                seal = read(directory/'validation_seal.json')
                require_digest(directory/'fit.json', seal['fit_sha256'])
                require_digest(directory/'val_results.json', seal['val_sha256'])
        for name, info in cfg['sources'].items():
            source, out = Path(info['path']), base/name
            for task in tasks:
                extract(source, out/'features', split, task, smoke=smoke, **cfg['extraction'])
            evaluate(source, out/'features', out/'readout', split, tasks, fitting, smoke)
        for task in tasks:
            with np.load(base/'equal90'/'readout'/split/f'task{task}.npz', allow_pickle=False) as a, \
                    np.load(base/'full'/'readout'/split/f'task{task}.npz', allow_pickle=False) as b:
                for key in ('image_ids', 'person_ids', 'inverse', 'targets', 'reliable', 'counts', 'reliable_counts'):
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f'Unpaired {split} stage{task}: {key}')
    rows = []
    for name, info in cfg['sources'].items():
        reports = {s: read(base/name/'readout'/f'{s}_results.json') for s in ('val', 'test')}
        for method in METHODS:
            row = dict(coverage=name, method=method, name=NAMES[method], exposure=info['exposure'])
            if not smoke:
                row.update({s: {k: reports[s]['summary'][method]['all'][k] for k in KEYS} for s in reports})
                row['comparisons'] = {s: reports[s]['comparisons'][method] for s in reports}
            rows.append(row)
    write_verified(base/'comparison.json', dict(smoke=smoke, tasks=tasks, rows=rows, target_comparable=not smoke,
        metric='final: stage7 all26 mAP; average: arithmetic mean of eight all-seen stage mAPs',
        original_D=BENCHMARK, evaluation=SCOPE))
    write_verified(base/'complete.json', dict(digest=record['digest'], sources=record['sources'],
                    comparison_sha256=sha256_file(base/'comparison.json')))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--config', type=Path, default=CONFIG)
    cli.add_argument('--phase', choices=('check', 'preflight', 'formal'), required=True)
    args = cli.parse_args()
    if Path.cwd().resolve() != ROOT:
        cli.error('Run from the Linux experiment worktree root')
    cfg = read(args.config)
    state = dict(phase=args.phase, status='running', started=datetime.now(timezone.utc).isoformat())
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    write_json(output/'run_state.json', state)
    try:
        run(cfg, args.phase)
    except BaseException as error:
        state.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic()-start)
        write_json(output/'run_state.json', state)
        raise
    state.update(status='complete', elapsed_seconds=time.monotonic()-start, finished=datetime.now(timezone.utc).isoformat())
    write_json(output/'run_state.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
