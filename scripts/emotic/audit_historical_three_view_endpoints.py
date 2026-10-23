"""Read-only, fixed-weight endpoint attribution; never selects a test rule."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_validation_scores import validated_run_scores
from multi_lane.track_a.runner import compute_metrics, summarize_tasks


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ('full', 'person', 'face')
    sources = dict(zip(names, (args.full, args.person, args.face)))
    old = {name: validated_run_scores(path, 'test')[0] for name, path in sources.items()}
    reliability = load_face_reliability(args.face_manifest_root, 'test')
    rows = {}
    checked = []
    files = []
    for task in range(8):
        with np.load(args.shared / 'view_scores' / f'task{task}.npz') as data:
            keep = (data['targets'] == 1).any(axis=1)
            ids, targets = data['sample_ids'][keep], data['targets'][keep]
            assert len(set(ids.tolist())) == len(ids)
            reliable = data['face_reliable'][keep]
            assert np.array_equal(reliable, [reliability[str(i)] for i in ids])
            weights = np.tile(np.asarray([.8, .2, 0], np.float32), (len(ids), 1))
            weights[reliable] = np.asarray([.64, .16, .2], np.float32)
            arrays = {'independent': {}, 'shared': {}}
            for name in names:
                dump = old[name][task]
                positions = {str(i): j for j, i in enumerate(dump.sample_ids)}
                assert set(positions) == set(ids.tolist())
                order = np.asarray([positions[str(i)] for i in ids])
                assert np.array_equal(targets, dump.targets[order])
                assert np.array_equal(data['class_indices'], dump.class_indices)
                arrays['independent'][name] = dump.probabilities[order]
                arrays['shared'][name] = data[name + '_probabilities'][keep]
                checked.append(dict(task=task, view=name, samples=len(ids)))
            scores = {'shared_trained_feature_fusion': data['fused_probabilities'][keep]}
            for source in arrays:
                for name in names:
                    scores[source + '_' + name] = arrays[source][name]
            # Enumerate the complete prespecified 2^3 attribution, without ranking
            # configurations or using these test values to select a future rule.
            for combination in itertools.product(('shared', 'independent'), repeat=3):
                key = 'probability_' + '_'.join(combination)
                scores[key] = sum(weights[:, i, None] * arrays[source][name]
                                  for i, (name, source) in enumerate(zip(names, combination)))
            for key, probabilities in scores.items():
                rows.setdefault(key, []).append(compute_metrics(
                    task, torch.from_numpy(probabilities), torch.from_numpy(targets), .5))
        paths = [args.shared / 'view_scores' / f'task{task}.npz']
        paths += [source / 'test_scores' / f'task{task}.npz' for source in sources.values()]
        for path in paths:
            files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary = {key: summarize_tasks(value) for key, value in rows.items()}
    assert abs(summary['probability_independent_independent_independent']['final_mAP'] - 33.339154) < 1e-5
    assert abs(summary['independent_full']['final_mAP'] - 32.53651921448899) < 1e-8
    assert abs(summary['shared_trained_feature_fusion']['final_mAP'] - 32.16384952422695) < 1e-8
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(dict(
        status='verified', evaluation_scope='samples_intersect_seen_classes', seed=0,
        training=False, model_forward=False, parameters_changed=False, test_rule_selection=False,
        purpose='Complete fixed-weight endpoint substitution for attribution only',
        reliability_mask_matches=True, people_labels_classes_checks=checked,
        independent_sources={k: str(v) for k, v in sources.items()},
        shared_source=str(args.shared), input_sha256=files, metrics=summary), indent=2) + '\n')
    fields = ['condition', 'final_mAP', 'average_mAP', 'final_cF1', 'final_oF1']
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for key, values in summary.items():
            row = dict(condition=key, **{field: values[field] for field in fields[1:]})
            writer.writerow(row)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['full', 'person', 'face', 'shared', 'face-manifest-root', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
