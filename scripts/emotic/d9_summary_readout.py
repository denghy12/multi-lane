"""Identical TRAIN-only fitting and paired evaluation for both branch coverages."""
from dataclasses import dataclass, replace
import hashlib
import os

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, average_precision, task_indices
from multi_lane.track_a.summary_logit_fusion import (
    METHODS, DIMENSIONS, SummaryLogitGate, fused_logits, image_bce, masked_weights,
)
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import native_scores, assert_train_roles
from scripts.emotic.d9_image_gate_readout import image_rows
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_extract import SCHEMA
from scripts.emotic.r1_asymmetric_forward import fuse_logits


@dataclass
class Stage:
    ids: np.ndarray
    person_ids: np.ndarray
    descriptors: dict
    counts: np.ndarray
    reliable_counts: np.ndarray
    inverse: torch.Tensor
    reliable: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    provenance: dict


def load_stage(source, features, split, task, smoke):
    path, meta, digest, checkpoint = native_scores(source, features, split, task)
    p = meta['protocol']
    if p['task'] != task or p['split'] != split or p['smoke']:
        raise ValueError('Expected formal archived native scores')
    training = split == 'train'
    if training and p['classes'] != list(task_indices(task)):
        raise ValueError('TRAIN class order changed')
    if training:
        for key, relative in (('config', 'config.json'), ('roles', 'train_image_roles.json'),
                              ('cohort', f'cohorts/task{task}.json')):
            if p[key+'_sha256'] != sha256_file(source/relative):
                raise ValueError(f'TRAIN native {key} provenance changed')
    feat = features/split/f'task{task}.npz'
    fm = read(feat.with_suffix('.json'))
    expected = dict(schema=SCHEMA, task=task, split=split, smoke=smoke, native_sha256=digest,
                    checkpoint_sha256=checkpoint, config_sha256=sha256_file(source/'config.json'),
                    roles_sha256=sha256_file(source/'train_image_roles.json'))
    if any(fm['protocol'].get(k) != v for k, v in expected.items()):
        raise ValueError(f'Feature provenance mismatch: {feat}')
    require_digest(feat, fm['archive_sha256'])
    with np.load(feat, allow_pickle=False) as f, np.load(path, allow_pickle=False) as z:
        if training:
            cohort = read(source/'cohorts'/f'task{task}.json')['fusion']
            person_digest = hashlib.sha256('\n'.join(z['sample_ids'].tolist()).encode()).hexdigest()
            if (len(z['sample_ids']) != cohort['people'] or person_digest != cohort['sample_id_sha256']
                    or z['targets'].sum(0).astype(int).tolist() != cohort['positives']):
                raise ValueError('TRAIN archive differs from the original legal supervised cohort')
        ids = f['image_ids']
        keys = image_rows(z['sample_ids'], split)
        positions = {k: i for i, k in enumerate(ids.tolist())}
        take = np.isin(keys, ids)
        if (len(positions) != len(ids) or not len(ids) or set(keys[take]) != set(ids)
                or not (smoke or take.all())):
            raise ValueError('Image roster and supervised population differ')
        inverse = np.asarray([positions[k] for k in keys[take]], dtype=np.int64)
        reliable = z['reliable'][take]
        counts, rc = f['counts'], f['reliable_counts']
        if (reliable.dtype != np.bool_ or counts.shape != (len(ids),) or rc.shape != counts.shape
                or np.any(rc < 0) or np.any(counts < 1) or np.any(rc > counts)
                or np.any(rc < np.bincount(inverse[reliable], minlength=len(ids)))
                or np.any(counts-rc < np.bincount(inverse[~reliable], minlength=len(ids)))):
            raise ValueError('Invalid complete-roster Face availability')
        logits, targets = torch.from_numpy(z['branch_logits'][take]).float(), torch.from_numpy(z['targets'][take]).float()
        size = TASK_SIZES[task] if training else sum(TASK_SIZES[:task+1])
        descriptors = {m: torch.from_numpy(f[m]).float() for m in METHODS[2:]}
        descriptors.update({m: torch.empty(len(ids), 0) for m in METHODS[:2]})
        if (logits.shape != (len(inverse), size, 3) or targets.shape != logits.shape[:2]
                or not ((targets == 0) | (targets == 1)).all()
                or not all(torch.isfinite(x).all() for x in (logits, targets, *descriptors.values()))
                or any(x.shape != (len(ids), DIMENSIONS[m]) for m, x in descriptors.items())):
            raise ValueError('Invalid current labels, original logits or intermediate descriptors')
        if not training:
            if list(z['view_order']) != ['full', 'person', 'face'] or list(z['task_sizes']) != list(TASK_SIZES[:task+1]):
                raise ValueError('Evaluation head/view order changed')
            replay = fuse_logits(*logits.unbind(-1), torch.from_numpy(reliable))
            torch.testing.assert_close(replay, torch.from_numpy(z['fixed_logits'][take]), rtol=2e-6, atol=5e-6)
            # Pilot reports only newly introduced classes, never an eight-stage aggregate.
            logits, targets = logits[:, -TASK_SIZES[task]:], targets[:, -TASK_SIZES[task]:]
        person_ids = z['sample_ids'][take]
    if training:
        assert_train_roles(source, ids)
    provenance = dict(native_sha256=digest, features_sha256=fm['archive_sha256'],
                      checkpoint_sha256=checkpoint, config_sha256=expected['config_sha256'])
    return Stage(ids, person_ids, descriptors, counts, rc, torch.from_numpy(inverse),
                 torch.from_numpy(reliable), logits, targets, provenance)


def subset(stage, indices):
    indices = np.asarray(indices, dtype=np.int64)
    remap = torch.full((len(stage.ids),), -1, dtype=torch.long)
    remap[torch.from_numpy(indices)] = torch.arange(len(indices))
    take = remap[stage.inverse] >= 0
    return replace(stage, ids=stage.ids[indices], person_ids=stage.person_ids[take.numpy()],
                   descriptors={m: x[indices] for m, x in stage.descriptors.items()}, counts=stage.counts[indices],
                   reliable_counts=stage.reliable_counts[indices], inverse=remap[stage.inverse[take]],
                   reliable=stage.reliable[take], logits=stage.logits[take], targets=stage.targets[take])


def folds(ids, number=3):
    # The same original image keeps its fold across tasks and source coverages.
    return np.asarray([int(hashlib.sha256(('summary-fold-v1:'+str(k)).encode()).hexdigest(), 16) % number
                       for k in ids])


def fit_one(stage, method, config):
    gate = SummaryLogitGate(method, config['amplitude'])
    x = stage.descriptors[method]
    gate.set_center(x)
    def loss():
        logits = fused_logits(stage.logits, gate(x), stage.inverse, stage.reliable)
        return image_bce(logits, stage.targets, stage.inverse, len(stage.ids))
    history = [float(loss().detach())]
    if method != 'fixed':
        groups = [dict(params=[gate.bias], lr=config['bias_lr'])]
        if gate.matrix.numel():
            groups.append(dict(params=[gate.matrix], lr=config['matrix_lr']))
        optimizer = torch.optim.Adam(groups, weight_decay=0)
        for step in range(config['steps']):
            optimizer.zero_grad(set_to_none=True)
            bce = loss()
            penalty = config['matrix_l2'] * gate.matrix.square().mean() if gate.matrix.numel() else bce * 0
            total = bce + penalty
            total.backward()
            if not torch.isfinite(total) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                               for p in gate.parameters() if p.requires_grad):
                raise FloatingPointError('Nonfinite fusion loss or missing gate gradient')
            optimizer.step()
            if (step+1) % 25 == 0 or step+1 == config['steps']:
                history.append(float(loss().detach()))
    return dict(gate=gate.export(), bce_history=history,
                parameters=sum(p.numel() for p in gate.parameters() if p.requires_grad))


def predictions(stage, gate):
    with torch.no_grad():
        scores = gate(stage.descriptors[gate.method])
        logits = fused_logits(stage.logits, scores, stage.inverse, stage.reliable)
    return scores, logits


def metrics(logits, targets):
    p, y = logits.detach().sigmoid().numpy(), targets.numpy()
    ap = [100*average_precision(s, t) for s, t in zip(p.T, y.T)]
    return dict(mAP=float(np.mean(ap)), per_class_ap=ap, people=len(y),
                positives=y.sum(0).astype(int).tolist(), negatives=(1-y).sum(0).astype(int).tolist())


def coverage(stage):
    return dict(images=len(stage.ids), people=len(stage.inverse), complete_roster_people=int(stage.counts.sum()),
                reliable_faces=int(stage.reliable.sum()), mixed_face_images=int(((stage.reliable_counts>0) &
                (stage.reliable_counts<stage.counts)).sum()), positives=stage.targets.sum(0).int().tolist(),
                negatives=(1-stage.targets).sum(0).int().tolist())


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise ValueError(f'Archive schema differs: {path}')
            for key, value in arrays.items():
                np.testing.assert_array_equal(old[key], value)
        return
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def train(source, features, out, tasks, config, smoke):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    rows = []
    for task in tasks:
        stage = load_stage(source, features, 'train', task, smoke)
        assignment = folds(stage.ids, config['folds'])
        if set(assignment.tolist()) != set(range(config['folds'])):
            raise ValueError('Not enough original images for every TRAIN fold')
        states, oof = {}, {}
        for method in METHODS:
            fold_rows, fold_predictions = [], torch.empty_like(stage.targets)
            for fold in range(config['folds']):
                train_ids, held_ids = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
                fitting, held = subset(stage, train_ids), subset(stage, held_ids)
                state = fit_one(fitting, method, config)
                _, logits = predictions(held, SummaryLogitGate.restore(state['gate']))
                mask = torch.from_numpy(assignment == fold)[stage.inverse]
                fold_predictions[mask] = logits
                fold_rows.append(dict(fold=fold, train=coverage(fitting), held=coverage(held),
                                      metrics=metrics(logits, held.targets), fit=state))
            final = fit_one(stage, method, config)
            _, fitted = predictions(stage, SummaryLogitGate.restore(final['gate']))
            states[method] = dict(**final, train_metrics=metrics(fitted, stage.targets),
                                  out_of_fold=metrics(fold_predictions, stage.targets), folds=fold_rows)
            oof[method] = fold_predictions.numpy()
        oof_path = out/'train'/f'task{task}_oof.npz'
        save_npz(oof_path, image_ids=stage.ids, person_ids=stage.person_ids, inverse=stage.inverse.numpy(),
                 targets=stage.targets.numpy(), fold=assignment, **oof)
        rows.append(dict(task=task, coverage=coverage(stage), provenance=stage.provenance, methods=states,
                         oof_sha256=sha256_file(oof_path)))
        print(f'fit task{task}: {len(stage.ids)} images, four methods, three TRAIN folds', flush=True)
    write_verified(out/'fit.json', dict(tasks=tasks, config=config, smoke=smoke, stages=rows,
                   protocol='independent current-head pilots; complete-roster inputs; image-balanced TRAIN BCE'))


def availability(stage):
    return np.where(stage.reliable_counts == 0, 0, np.where(stage.reliable_counts == stage.counts, 2, 1))


def group_mean_scores(scores, stage, locked=None):
    kind = availability(stage)
    if locked is None:
        # No labels used. Includes a deterministic fallback for a group absent in Val.
        locked = dict(fallback=scores.mean(0).tolist(),
                      groups={str(k): scores[kind == k].mean(0).tolist() for k in (0, 1, 2) if (kind == k).any()})
    mean = torch.tensor([locked['groups'].get(str(k), locked['fallback']) for k in kind], dtype=scores.dtype)
    return mean, locked


def permutation(stage, rng):
    donors = np.arange(len(stage.ids))
    kind = availability(stage)
    for k in (0, 1, 2):
        rows = np.flatnonzero(kind == k)
        if len(rows) > 1:
            cycle = rng.permutation(rows)
            donors[cycle] = np.roll(cycle, 1)
    return torch.from_numpy(donors)


def grouped_metrics(stage, logits):
    owner = stage.inverse.numpy()
    groups = dict(all=np.ones(len(owner), dtype=bool), reliable=stage.reliable.numpy(),
                  unreliable=~stage.reliable.numpy(), single=stage.counts[owner] == 1,
                  multiple=stage.counts[owner] > 1, mixed_faces=availability(stage)[owner] == 1)
    return {k: metrics(logits[v], stage.targets[v]) for k, v in groups.items() if v.any()}


def evaluate(source, features, out, split, tasks, config, smoke):
    torch.set_num_threads(1)
    fit = read(out/'fit.json')
    if fit['tasks'] != tasks or fit['config'] != config or fit['smoke'] != smoke:
        raise ValueError('TRAIN fitting protocol differs from evaluation')
    seal_path = out/'validation_seal.json'
    seal = None
    if split == 'test':
        seal = read(seal_path)
        require_digest(out/'fit.json', seal['fit_sha256'])
        require_digest(out/'val_results.json', seal['val_sha256'])
    rows, mean_locks = [], {}
    for fitted in fit['stages']:
        task = fitted['task']
        stage = load_stage(source, features, split, task, smoke)
        for key in ('checkpoint_sha256', 'config_sha256'):
            if fitted['provenance'][key] != stage.provenance[key]:
                raise ValueError('Evaluation is not paired to fitted branches')
        result, arrays, task_locks = {}, {}, {}
        for method in METHODS:
            gate = SummaryLogitGate.restore(fitted['methods'][method]['gate'])
            scores, logits = predictions(stage, gate)
            if method == 'fixed':
                native_fixed = fuse_logits(*stage.logits.unbind(-1), stage.reliable)
                torch.testing.assert_close(logits, native_fixed, rtol=2e-6, atol=5e-6)
            result[method] = dict(groups=grouped_metrics(stage, logits))
            arrays[method+'_scores'], arrays[method+'_logits'] = scores.numpy(), logits.numpy()
            arrays[method+'_weights'] = masked_weights(scores[stage.inverse], stage.reliable).numpy()
            if method in METHODS[2:]:
                locked = None if seal is None else seal['mean_scores'][str(task)][method]
                mean_scores, task_locks[method] = group_mean_scores(scores, stage, locked)
                mean_logits = fused_logits(stage.logits, mean_scores, stage.inverse, stage.reliable)
                result[method]['mean_scores'] = grouped_metrics(stage, mean_logits)
                arrays[method+'_mean_logits'] = mean_logits.numpy()
                rng, repeats, perm_logits = np.random.default_rng(config['permutation_seed']), [], []
                for _ in range(config['permutations']):
                    shuffled = fused_logits(stage.logits, scores[permutation(stage, rng)], stage.inverse, stage.reliable)
                    repeats.append(metrics(shuffled, stage.targets)['mAP'])
                    perm_logits.append(shuffled.numpy())
                result[method]['permutation'] = dict(mAP=float(np.mean(repeats)),
                    std=float(np.std(repeats)), repetitions=len(repeats), values=repeats,
                    availability_group_sizes={str(k): int((availability(stage) == k).sum()) for k in (0, 1, 2)})
                arrays[method+'_permuted_logits'] = np.stack(perm_logits)
        archive = out/split/f'task{task}.npz'
        save_npz(archive, image_ids=stage.ids, person_ids=stage.person_ids, inverse=stage.inverse.numpy(),
                 targets=stage.targets.numpy(), reliable=stage.reliable.numpy(), counts=stage.counts,
                 reliable_counts=stage.reliable_counts, **arrays)
        rows.append(dict(task=task, coverage=coverage(stage), provenance=stage.provenance,
                         methods=result, predictions_sha256=sha256_file(archive)))
        mean_locks[str(task)] = task_locks
        print(f'{split} task{task}: native fixed replay, four methods, permutation controls', flush=True)
    result = dict(split=split, smoke=smoke, fit_sha256=sha256_file(out/'fit.json'), stages=rows,
                  metric='mean current-head mAP over independent pilot tasks; NOT final/average eight-task mAP',
                  mean_current_head_mAP={m: float(np.mean([r['methods'][m]['groups']['all']['mAP'] for r in rows]))
                                        for m in METHODS})
    write_verified(out/f'{split}_results.json', result)
    if split == 'val':
        write_verified(seal_path, dict(fit_sha256=sha256_file(out/'fit.json'),
                       val_sha256=sha256_file(out/'val_results.json'), mean_scores=mean_locks))
    return result
