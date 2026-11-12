"""Real image forward: original full-TRAIN Frozen9 with locked TRAIN BCE99 fusion."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.r1_crossvit_experiment import model_from_config, read, sha, write
from multi_lane.track_a.runner import git_metadata, set_seed, task_indices
from scripts.emotic.r1_frozen9_edge_forward import ROOT as EDGE_ROOT, branch_paths
from scripts.emotic.r1_full_train_fusion_weights import locked_weights, task_blend, align_exact
from scripts.emotic.run_r1_frozen9_learned_fusion import (
    ARCHIVE, BRANCHES, FIXED, archive_files, blend, export_rows, verify_hashes,
)
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, score_suffix
from scripts.emotic.r1_asymmetric_forward import verify_replay

ROOT = Path('output/r1_full_train_fusion_verify/seed1_20261112')
HOLDOUT = Path('../multi-lane-main-r1-train-holdout-fusion-20261111/output/r1_train_holdout_fusion/seed1_20261111')
EXPECTED = {'val': (44.1541, 51.2067), 'test': (33.2757, 40.2611)}


def run(root, holdout, smoke):
    # Never resume from cached predictions: this entry always executes image forwards.
    if root.exists():
        raise FileExistsError('Use a fresh output directory; preserve previous runs')
    if not torch.cuda.is_available():
        raise RuntimeError('Real image replay requires the remote CUDA environment')
    inputs = input_paths()
    paths = {name: path for name, path in branch_paths(EDGE_ROOT, 'readonly').items() if name in BRANCHES}
    if tuple(paths) != BRANCHES:
        raise AssertionError('View order changed')
    tasks = (0, 2, 7) if smoke else tuple(range(8))
    fit_path = holdout/'fit_report.json'
    files = [fit_path, inputs.clip_checkpoint]
    for path in paths.values():
        files.append(path/'config.json')
        files.extend(path/'compact_checkpoints'/f'task{t}.pth' for t in tasks)
    for split in ('val', 'test'):
        files.extend(archive_files(split, tasks))
    hashes = {str(p): sha(p) for p in files}
    weights = locked_weights(read(fit_path))
    # Lock TRAIN-derived parameters and protocol before any Val/Test forward.
    write(root/'locked_protocol.json', {
        'source': git_metadata(), 'seed': 1, 'tasks': tasks, 'smoke': smoke,
        'fit_split': 'train', 'optimization': False, 'source_step': 99,
        'batch_size': 64, 'workers': 2, 'cpu_threads': 2,
        'weights': [v.tolist() for v in weights], 'source_sha256': hashes,
        'expected_diagnostic': EXPECTED, 'expected_tolerance_map_points': .001,
        'purpose': 'cross-source diagnostic confirmation, not new formal training',
    })
    set_seed(1, True)
    torch.set_num_threads(2)
    models = {name: model_from_config(read(path/'config.json'), inputs.clip_checkpoint)
              for name, path in paths.items()}
    records = {s: make_records(('fixed', 'train_bce99')) for s in ('val', 'test')}
    replays = []
    for split in ('val', 'test'):
        source = load_source(paths['person_all'], inputs, split)
        for task in tasks:
            rows = export_rows(models, paths, source, split, task, smoke)
            suffix = score_suffix(split, task)
            for view, name in enumerate(BRANCHES):
                with np.load(ARCHIVE/'branches'/name/suffix, allow_pickle=False) as d:
                    if not smoke and len(rows.ids) != len(d['sample_ids']):
                        raise AssertionError('Formal replay population is incomplete')
                    z = align_exact(rows.ids, rows.targets, d['sample_ids'], d['targets'], d['logits'])
                if not np.array_equal(rows.logits[:, view], z):
                    raise AssertionError(f'Branch replay mismatch: {split}/{task}/{name}')
                replays.append({'split': split, 'task': task, 'branch': name, 'max_abs': 0.})
            fused = {'fixed': blend(rows.logits, rows.reliable, FIXED),
                     'train_bce99': task_blend(rows.logits, rows.reliable, weights, task)}
            verify_replay(ARCHIVE/'fusion_scores/D_both_edges'/suffix,
                          rows.ids, torch.from_numpy(rows.targets), torch.from_numpy(fused['fixed']))
            record(records[split], task, fused, rows.targets, rows.reliable, list(task_indices(task)))
            p = root/'predictions'/split/f'task{task}.npz'
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, ids=rows.ids, targets=rows.targets, reliable=rows.reliable,
                                branch_logits=rows.logits, **fused)
            verify_hashes(hashes)
            write(root/'progress.json', {'split': split, 'task': task, 'replays': len(replays)})
            print('REAL_FORWARD', split, task, len(rows.ids), flush=True)
    result = {s: summarize(r, smoke) for s, r in records.items()}
    if not smoke:
        for split, expected in EXPECTED.items():
            values = result[split]['train_bce99']['all']['summary']
            for key, value in zip(('final_mAP', 'average_mAP'), expected):
                if abs(values[key]-value) > .001:
                    raise AssertionError('Real forward differs from the declared cached diagnostic')
    verify_hashes(hashes)
    write(root/'result.json', {'results': result, 'replays': replays})
    write(root/'completion.json', {'status': 'complete', 'smoke': smoke,
                                 'all_source_logits_exact': True, 'no_training': True,
                                 'ready_for_predeclared_training': not smoke})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--holdout-root', type=Path, default=HOLDOUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.root, args.holdout_root, args.smoke)
