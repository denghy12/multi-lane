"""Four intermediate-input logit-fusion controls x two TRAIN coverages (task0/3/6)."""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.summary_logit_fusion import METHODS, NAMES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import extract_train, source_mode
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_extract import extract
from scripts.emotic.d9_summary_readout import train, evaluate
from scripts.emotic.d9_summary_diagnostics import bootstrap, decision
from scripts.emotic.evaluate_d9_relation import write_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'scripts/emotic/configs/d9_summary_fusion_20261126.json'
CODE = ('multi_lane/track_a/model.py', 'multi_lane/track_a/r1_crossvit.py',
        'multi_lane/track_a/r1_frozen_cross_view.py', 'multi_lane/track_a/d9_relation_fusion.py',
        'multi_lane/track_a/summary_logit_fusion.py', 'scripts/emotic/d9_summary_extract.py',
        'scripts/emotic/d9_summary_readout.py', 'scripts/emotic/d9_summary_diagnostics.py',
        'scripts/emotic/d9_summary_run.py')


def config_check(cfg):
    if (cfg['schema'] != 1 or cfg['seed'] != 1 or cfg['tasks'] != [0, 3, 6]
            or cfg['methods'] != list(METHODS) or set(cfg['sources']) != {'equal90', 'full'}):
        raise ValueError('Expected the declared eight-row, seed1 current-head pilot')
    if any(set(v) != {'path', 'mode', 'exposure'} for v in cfg['sources'].values()):
        raise ValueError('Only source/exposure may differ; no per-coverage hyperparameter overrides')
    expected = dict(steps=300, bias_lr=.01, matrix_lr=.001, amplitude=.25, matrix_l2=.001,
                    folds=3, permutations=100, permutation_seed=20261126,
                    bootstrap=1000, bootstrap_seed=20261127)
    if cfg['fit'] != expected or cfg['extraction'] != dict(batch_size=64, workers=2):
        raise ValueError('Locked protocol changed; define and document a new experiment config')
    if any(Path(v).is_absolute() or '..' in Path(v).parts for v in (cfg['output'], cfg['logs'])):
        raise ValueError('Results/logs must stay in project-relative output/logs directories')
    if not cfg['output'].startswith('output/') or not cfg['logs'].startswith('logs/'):
        raise ValueError('Use project output/ and logs/')
    if (cfg['sources']['equal90']['mode'] != 'sample_holdout'
            or cfg['sources']['full']['mode'] != 'sample_full'):
        raise ValueError('Coverage labels must match branch source modes')


def source_check(cfg):
    configurations, records = {}, {}
    for name, info in cfg['sources'].items():
        source = Path(info['path'])
        c = read(source/'config.json')
        if source_mode(source) != info['mode']:
            raise ValueError(f'Unexpected branch source: {name}')
        if c['fusion_preprocessing'] != 'deterministic_train_images' or c['epochs'] != 30 or c['tasks'] != 8:
            raise ValueError('Unexpected source preprocessing/training budget')
        if name == 'equal90':
            equal = c.get('equal_updates', {})
            if (equal.get('mode') != 'cycle_90pct_to_full_epoch_size'
                    or sum(v['updates'] for v in equal['tasks'].values()) * c['epochs'] != 13950):
                raise ValueError('Requires the original equal-update 90% branches')
        configurations[name] = c
        records[name] = dict(mode=info['mode'], exposure=info['exposure'],
            config_sha256=sha256_file(source/'config.json'), roles_sha256=sha256_file(source/'train_image_roles.json'),
            checkpoints={str(t): sha256_file(source/'compact_checkpoints'/f'task{t}.pth') for t in cfg['tasks']})
    for key in ('branches', 'seed', 'epochs', 'precision', 'clip_checkpoint', 'protect_branches'):
        if configurations['equal90'][key] != configurations['full'][key]:
            raise ValueError(f'Branch source configuration mismatch: {key}')
    roles = {k: read(Path(v['path'])/'train_image_roles.json') for k, v in cfg['sources'].items()}
    if (set(roles['equal90']) != set(roles['full']) or set(roles['full'].values()) != {'both'}
            or list(roles['equal90'].values()).count('fusion') != 1292
            or list(roles['equal90'].values()).count('branch') != 11623):
        raise ValueError('Expected the original 11623/1292 image split and matching full TRAIN pool')
    return records


def code_record():
    files = dict.fromkeys(CODE)
    # Include every tracked Python dependency: old source/roster helpers affect provenance too.
    listed = subprocess.check_output(['git', 'ls-files', 'multi_lane', 'scripts/emotic', 'tests'], cwd=ROOT, text=True)
    files.update({p: None for p in listed.splitlines() if p.endswith('.py')})
    return {p: sha256_file(ROOT/p) for p in sorted(files)}


def protocol(cfg):
    files = code_record()
    digest = hashlib.sha256(json.dumps(dict(config=cfg, files=files), sort_keys=True).encode()).hexdigest()
    return dict(digest=digest, code_sha256=files, config=cfg, sources=source_check(cfg),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def run(cfg, phase):
    config_check(cfg)
    smoke = phase == 'preflight'
    record = protocol(cfg)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit tracked implementation/config changes before a reproducible Linux run')
    base = Path(cfg['output'])/('preflight' if smoke else 'pilot')
    if not smoke:
        prior = read(Path(cfg['output'])/'preflight'/'complete.json')
        if prior['digest'] != record['digest'] or prior['sources'] != record['sources']:
            raise ValueError('This code/config/source combination has not passed preflight')
    base.mkdir(parents=True, exist_ok=True)
    write_verified(base/'protocol.json', record)
    tasks = [0, 6] if smoke else cfg['tasks']
    fitting = dict(cfg['fit'])
    if smoke:
        fitting.update(steps=5, permutations=2, bootstrap=10)
        subprocess.run(['python', '-m', 'unittest', 'discover', '-s', 'tests',
                        '-p', 'test_d9_summary*.py', '-v'], cwd=ROOT, check=True)
    for name, info in cfg['sources'].items():
        source, out = Path(info['path']), base/name
        if info['mode'] == 'sample_full':
            for task in tasks:
                extract_train(source, out/'native_train', task, **cfg['extraction'])
        for task in tasks:
            extract(source, out/'features', 'train', task, smoke=smoke, **cfg['extraction'])
        train(source, out/'features', out/'readout', tasks, fitting, smoke)
    # Both coverages finish TRAIN and Val before either Test is accessed.
    for split in ('val', 'test'):
        if split == 'test':
            for name in cfg['sources']:
                seal = read(base/name/'readout'/'validation_decision_seal.json')
                require_digest(base/name/'readout'/'validation_decision.json', seal['decision_sha256'])
                require_digest(base/name/'readout'/'val_bootstrap.json', seal['bootstrap_sha256'])
                require_digest(base/name/'readout'/'validation_seal.json', seal['score_seal_sha256'])
        for name, info in cfg['sources'].items():
            source, out = Path(info['path']), base/name
            for task in tasks:
                extract(source, out/'features', split, task, smoke=smoke, **cfg['extraction'])
            evaluate(source, out/'features', out/'readout', split, tasks, fitting, smoke)
            bootstrap(out/'readout', split, fitting['bootstrap'], fitting['bootstrap_seed'])
            if split == 'val':
                decision(out/'readout')
                # A separate, immutable decision seal extends the readout's score seal.
                seal_path = out/'readout'/'validation_decision_seal.json'
                write_verified(seal_path, dict(decision_sha256=sha256_file(out/'readout'/'validation_decision.json'),
                    bootstrap_sha256=sha256_file(out/'readout'/'val_bootstrap.json'),
                    score_seal_sha256=sha256_file(out/'readout'/'validation_seal.json')))
        for task in tasks:
            with np.load(base/'equal90'/'readout'/split/f'task{task}.npz', allow_pickle=False) as a, \
                    np.load(base/'full'/'readout'/split/f'task{task}.npz', allow_pickle=False) as b:
                for key in ('image_ids', 'person_ids', 'inverse', 'targets', 'reliable', 'counts', 'reliable_counts'):
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f'Unpaired {split} population: {key}')
    table = []
    for name, info in cfg['sources'].items():
        readout = base/name/'readout'
        val, test = read(readout/'val_results.json'), read(readout/'test_results.json')
        for m in METHODS:
            table.append(dict(coverage=name, exposure=info['exposure'], method=m, name=NAMES[m],
                         val=val['mean_current_head_mAP'][m], test=test['mean_current_head_mAP'][m]))
    write_verified(base/'comparison.json', dict(smoke=smoke, tasks=tasks, rows=table,
        metric='independent current-head pilot mean mAP, NOT eight-task final/average',
        original_eight_task_target=dict(val=[44.0329, 51.1335], test=[33.1770, 40.2083]),
        target_comparable=False))
    write_verified(base/'complete.json', dict(digest=record['digest'], sources=record['sources'],
                    comparison_sha256=sha256_file(base/'comparison.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--phase', choices=('check', 'preflight', 'pilot'), required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('Run from the Linux experiment worktree root')
    cfg = read(args.config)
    config_check(cfg)
    if args.phase == 'check':
        print(json.dumps(protocol(cfg), indent=2, ensure_ascii=False))
    else:
        output = Path(cfg['output'])
        output.mkdir(parents=True, exist_ok=True)
        state = dict(phase=args.phase, status='running',
                     started=datetime.now(timezone.utc).isoformat())
        write_json(output/'run_state.json', state)
        start = time.monotonic()
        try:
            run(cfg, args.phase)
        except BaseException as error:
            state.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic()-start)
            write_json(output/'run_state.json', state)
            raise
        state.update(status='complete', elapsed_seconds=time.monotonic()-start,
                     finished=datetime.now(timezone.utc).isoformat())
        write_json(output/'run_state.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
