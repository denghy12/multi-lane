"""Measure the value of assigning one of three fixed fusion actions to each person and task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, batch_map, permutation_sources
from scripts.emotic.evaluate_d9_relation import write_json


def action_data(data, sizes):
    fixed = data['fixed_logits']
    branches = data['branch_logits']
    labels = data['targets']
    reliable = data['reliable']
    columns = np.repeat(np.arange(len(sizes)), sizes)
    candidates = np.stack([fixed, fixed + .05 * (branches[..., 1] - branches[..., 0]),
                           fixed + .05 * (branches[..., 2] - branches[..., 0])], axis=-1)
    candidates[~reliable, :, 2] = fixed[~reliable]
    if not np.isfinite(candidates).all():
        raise ValueError('Nonfinite candidate')
    scores = torch.from_numpy(candidates)
    truth = torch.from_numpy(labels).float().unsqueeze(-1).expand_as(scores)
    loss = F.binary_cross_entropy_with_logits(scores, truth, reduction='none')
    losses = np.stack([block.mean(1).numpy() for block in loss.split(sizes, dim=1)], axis=1)
    gain = losses[..., :1] - losses
    # Separate CPU/GPU float32 reductions can differ in the last bits.
    np.testing.assert_allclose(gain[..., 1:], data['direction_bce_reduction'], atol=1e-5, rtol=1e-5)
    gain[~reliable, :, 2] = -np.inf
    return candidates, columns, gain, losses[:, :, 0]


def evaluate_stage(data, stage, report, locked):
    labels, ids, reliable = data['targets'], data['sample_ids'], data['reliable']
    sizes = data['task_sizes'].tolist()
    if sizes != list(TASK_SIZES[:stage + 1]) or data['view_order'].tolist() != ['full', 'person', 'face']:
        raise ValueError('Unexpected class or view order')
    candidates, columns, gain, base_loss = action_data(data, sizes)
    n, h = base_loss.shape
    own = gain.argmax(axis=2)  # tie priority: keep, Person, Face
    means = {str(int(flag)): gain[reliable == flag].mean(0).argmax(1).tolist() for flag in (False, True)}

    def assign(center):
        return np.where(reliable[:, None], center['1'], center['0'])

    def outputs(actions):
        if (actions[~reliable] == 2).any():
            raise ValueError('Unavailable Face was selected')
        scores = np.take_along_axis(candidates, actions[:, columns, None], axis=2)[:, :, 0]
        delta = np.take_along_axis(gain, actions[:, :, None], axis=2)[:, :, 0]
        return scores, delta

    def measure(actions):
        scores, delta = outputs(actions)
        row = {}
        for group, mask in [('all', np.ones(n, bool)), ('reliable', reliable), ('unreliable', ~reliable)]:
            row[group] = dict(people=int(mask.sum()), map=float(batch_map(scores[None, mask], labels[mask])[0]),
                              bce=float((base_loss[mask] - delta[mask]).mean(dtype=np.float64)),
                              action_fractions=(np.bincount(actions[mask].ravel(), minlength=3) /
                                                actions[mask].size).tolist())
        return row

    keep = np.zeros_like(own)
    arms = dict(fixed=keep, person_all=keep + 1, face_if_reliable=keep + 2 * reliable[:, None],
                group_same_split=assign(means), group_val_locked=assign(locked or means), person_oracle=own)
    metrics = {k: measure(a) for k, a in arms.items()}
    for g, values in metrics['fixed'].items():
        np.testing.assert_allclose(values['map'], report['metrics']['fixed'][g]['mAP'], atol=1e-10, rtol=0)
    _, own_gain = outputs(own)
    if (own_gain < 0).any():
        raise ValueError('Oracle omitted keep')
    group_gain = outputs(arms['group_same_split'])[1]
    if (own_gain < group_gain).any():
        raise ValueError('Inconsistent per-person selection')
    shuffle = []
    for seed in FORMAL_SEEDS:
        donors = permutation_sources(ids, reliable, h, seed)
        shuffled = own[donors, np.arange(h)]
        for flag in (False, True):
            mask = reliable == flag
            for a in range(3):
                np.testing.assert_array_equal((own[mask] == a).sum(0), (shuffled[mask] == a).sum(0))
        shuffle.append(measure(shuffled))
    new = columns == stage
    new_class_map = {}
    for k, a in arms.items():
        scores = outputs(a)[0][:, new]
        new_class_map[k] = float(batch_map(scores[None], labels[:, new])[0])
    return dict(stage=stage, group_actions=means, locked_group_actions=locked or means,
                metrics=metrics, new_class_map=new_class_map, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(exist_ok=True, parents=True)
    protocol = dict(actions=['keep', 'Full to Person 0.05', 'Full to reliable Face 0.05'],
                    selection='minimum mean BCE across classes of each task head per person; exact ties prefer keep then Person',
                    group='stage x task head x Face reliability; split-specific BCE optimum plus Validation-locked control',
                    seed=1, permutation_seeds=list(FORMAL_SEEDS),
                    bce='equal weight per person and per seen task head within stage',
                    purpose='Uses true labels for oracle assignment in each split. No fitting or model updates.')
    write_json(args.output/'protocol.json', protocol)
    validation = []
    for split in ('val', 'test'):
        root = args.source/'evaluation'/split
        report = json.loads((root/'result.json').read_text())
        if report['smoke'] or report['split'] != split or len(report['stages']) != len(TASK_SIZES):
            raise ValueError('Expected complete official evaluation')
        stages = []
        for i, row in enumerate(report['stages']):
            archive = root/f'task{i}.npz'
            if not row['exact_replay'] or sha256_file(archive) != row['scores_sha256']:
                raise ValueError(f'Archive mismatch at {archive}')
            with np.load(archive, allow_pickle=False) as data:
                result = evaluate_stage(data, i, row, validation[i] if split == 'test' else None)
            result['source_sha256'] = row['scores_sha256']
            stages.append(result)
            if split == 'val':
                validation.append(result['group_actions'])
        summary = {}
        for g in ('all', 'reliable', 'unreliable'):
            summary[g] = {}
            for name in stages[0]['metrics']:
                values = [r['metrics'][name][g]['map'] for r in stages]
                summary[g][name] = [values[-1], float(np.mean(values))]
            sh = np.array([[v[g]['map'] for v in r['shuffle']] for r in stages])
            summary[g]['shuffle_mean'] = [float(sh[-1].mean()), float(sh.mean())]
            summary[g]['shuffle_range_final_avg'] = [[float(x.min()), float(x.max())] for x in (sh[-1], sh.mean(0))]
        write_json(args.output/f'{split}.json', dict(protocol=protocol, stages=stages, summary=summary))
        print(split, json.dumps(summary['all']), flush=True)


if __name__ == '__main__':
    main()
