"""Offline paired validation attribution for co-trained joint-summary residuals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def ap(scores, labels):
    y = labels[scores.argsort()[::-1]] == 1
    return float((np.cumsum(y) * y / np.arange(1, len(y)+1)).sum() / (y.sum()+1e-8)*100)


def map_score(scores, labels):
    return float(np.mean([ap(scores[:, c], labels[:, c]) for c in range(labels.shape[1])]))


def analyze(root, reference, replicates=500):
    result = dict(split='val', updates=0, test_used=False,
                  uncertainty='Original-image sampling conditional on these fixed seed0 checkpoints; not training-seed uncertainty.',
                  interpretation='off retains co-trained Selectors, Prompts and head; it is not a retrained fixed baseline.',
                  arms={})
    for arm in ('dense', 'moe'):
        folder = root/arm
        on_info = json.loads((folder/'on/result.json').read_text())
        off_info = json.loads((folder/'off/result.json').read_text())
        assert on_info['checkpoint_sha256'] == off_info['checkpoint_sha256']
        assert on_info['checkpoint_tasks'] == off_info['checkpoint_tasks']
        assert on_info['joint_residual_runtime_enabled'] and not off_info['joint_residual_runtime_enabled']
        replay = json.loads((folder/'on/original_replay.json').read_text())
        rows = []
        for task in on_info['checkpoint_tasks']:
            on = np.load(folder/f'on/scores/task{task}.npz')
            off = np.load(folder/f'off/scores/task{task}.npz')
            face = np.load(reference/f'view_scores/task{task}.npz')
            for other in (off, face):
                assert np.array_equal(on['sample_ids'], other['sample_ids'])
                assert np.array_equal(on['targets'], other['targets'])
            assert replay[str(task)] == 'all_arrays_bitwise_equal'
            p, q, y = on['probabilities'], off['probabilities'], on['targets']
            rows.append(dict(task=task, on=map_score(p,y), off=map_score(q,y),
                             off_minus_on=map_score(q,y)-map_score(p,y),
                             max_logit_change=float(np.max(np.abs(on['logits']-off['logits']))), groups={}))
            # Shared original-image draws for all tasks, arms and Face subsets.
            image_ids = np.array([str(s).rsplit('#person=',1)[0] for s in on['sample_ids']])
            unique, groups = np.unique(image_ids, return_inverse=True)
            rng = np.random.default_rng(20261020)
            weights = [np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
                       for _ in range(replicates)]
            masks = dict(all=np.ones(len(y),dtype=bool), reliable_face=face['face_reliable'],
                         unreliable_face=~face['face_reliable'])
            for name, mask in masks.items():
                a, b, targets, g = p[mask], q[mask], y[mask], groups[mask]
                cache = []
                for scores in (a, b):
                    cache.append([(g[order], targets[order,c] == 1)
                        for c in range(y.shape[1]) for order in [scores[:,c].argsort()[::-1]]])
                draws = []
                for count in weights:
                    maps = []
                    for cols in cache:
                        aps = []
                        for gids, labels in cols:
                            labels = np.repeat(labels,count[gids])
                            aps.append((np.cumsum(labels)*labels/np.arange(1,len(labels)+1)).sum()
                                       / (labels.sum()+1e-8)*100)
                        maps.append(np.mean(aps))
                    draws.append(maps[1]-maps[0])
                rows[-1]['groups'][name] = dict(people=int(mask.sum()),
                    on=map_score(a,targets), off=map_score(b,targets),
                    off_minus_on=map_score(b,targets)-map_score(a,targets),
                    CI95=np.quantile(draws,[.025,.975]).tolist())
            print(arm,task,rows[-1]['on'],rows[-1]['off'],flush=True)
        result['arms'][arm] = dict(checkpoint_sha256=on_info['checkpoint_sha256'],
                                  source_git=on_info['source_git'],evaluation_git=on_info['evaluation_git'], rows=rows)
    path = root/'analysis.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--face-reference',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=500)
    args = parser.parse_args()
    analyze(args.root,args.face_reference,args.replicates)
