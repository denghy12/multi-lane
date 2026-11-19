"""Fit task-private or class-specific fusion on original TRAIN holdout people."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_fit import check_cohort
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import LR, fit_one, weights as shared_weights

METHODS = ('A', 'class_static', 'person_task', 'person_class')
VIEWS = ('full', 'person', 'face')
STEPS = 300
PROTOCOL = dict(version=1, tasks=list(range(len(TASK_SIZES))), methods=list(METHODS),
                seed=1, steps=STEPS, learning_rate=LR, optimizer='Adam',
                initialization='all zero => available views equal', loss='full current-task held-out TRAIN BCE',
                features='full concatenated classifier inputs; no learned or fixed compression; invalid Face zero',
                group_bias='independent unreliable/reliable bias; person FC coefficients shared between groups',
                freeze='branch logits, branch features, and all earlier fusion heads')


def read(path):
    return json.loads(path.read_text())


def design(cls, reliable):
    """Exclude invalid Face content before the FC, in addition to masking its output probability."""
    return torch.cat((cls[:, 0], cls[:, 1], cls[:, 2].masked_fill(~reliable[:, None], 0)), 1).float()


def initialize(method, classes, feature_dim):
    if method not in METHODS:
        raise ValueError(f'Unknown fusion method {method}')
    rows = classes if method in ('class_static', 'person_class') else 1
    model = dict(bias=torch.zeros(2, rows, 3))
    if method in ('person_task', 'person_class'):
        model['fc'] = torch.zeros(rows * 3, feature_dim)
    return model


def predict(model, x, reliable, n_classes):
    bias = model['bias']
    rows = bias.shape[1]
    if rows == 1 and 'fc' not in model:
        # Keep the native A arithmetic including the softmax kernel shape.
        return shared_weights(bias[:, 0], reliable)[:, None, :].expand(-1, n_classes, -1)
    if rows not in (1, n_classes):
        raise ValueError('Class dimension differs from the sealed task')
    s = bias[reliable.long()]
    if 'fc' in model:
        s = s + F.linear(x, model['fc']).reshape(-1, rows, 3)
    s = torch.stack((s[..., 0], s[..., 1], s[..., 2].masked_fill(~reliable[:, None], -torch.inf)), -1)
    return s.softmax(-1).expand(-1, n_classes, -1)


def readout_weights(model, cls, reliable, n_classes):
    """Return N x classes x 3 probabilities, matching the full/person/face view order."""
    return predict(model, design(cls, reliable) if 'fc' in model else None, reliable, n_classes)


def fuse(w, logits):
    return w[..., 0] * logits[..., 0] + w[..., 1] * logits[..., 1] + w[..., 2] * logits[..., 2]


def train_head(method, d, cls, reliable, targets, logits, steps=STEPS):
    model = initialize(method, targets.shape[1], cls.shape[1] * cls.shape[2])
    x = design(cls, reliable) if 'fc' in model else None

    def objective():
        w = predict(model, x, reliable, targets.shape[1])
        return w, F.binary_cross_entropy_with_logits(fuse(w, logits), targets)

    with torch.no_grad():
        _, before = objective()
    if method == 'A':
        model['bias'] = torch.tensor(fit_one(d, None, steps))[:, None, :]
    else:
        for p in model.values():
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(list(model.values()), lr=LR)
        for _ in range(steps):
            _, loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {method} fitting objective')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model = {k: p.detach() for k, p in model.items()}
    w, after = objective()
    if (not torch.isfinite(after) or not torch.isfinite(w).all() or w[~reliable, :, 2].count_nonzero()):
        raise FloatingPointError('Invalid fitted view probabilities')
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]), atol=1e-6, rtol=0)
    diagnostics = dict(initial_bce=before.item(), final_bce=after.item(),
                       parameters=sum(p.numel() for p in model.values()),
                       near_one_fraction=float((w.amax(dim=-1) > 0.99).float().mean()),
                       mean_weights_by_group=[w[reliable == flag].mean(0).tolist() for flag in (False, True)])
    return model, diagnostics


def model_path(out, task):
    return out / f'task{task}_fusion.npz'


def load_models(out):
    """Validate the sealed TRAIN artifacts, returning (lock, per-task per-method CPU models)."""
    out = Path(out)
    lock = read(out / 'caran_fit_lock.json')
    if lock['protocol'] != PROTOCOL or set(lock['tasks']) != {str(t) for t in PROTOCOL['tasks']}:
        raise ValueError('Incomplete or incompatible TRAIN lock')
    models = {}
    for t in PROTOCOL['tasks']:
        path = model_path(out, t)
        row = lock['tasks'][str(t)]
        if row['model_sha256'] != sha256_file(path):
            raise ValueError(f'Task {t} sealed fusion parameters changed')
        with np.load(path, allow_pickle=False) as archive:
            models[t] = {}
            expected = set()
            for name in METHODS:
                template = initialize(name, TASK_SIZES[t], 3 * row['head_cls_dim'])
                fitted = {}
                for key, p in template.items():
                    member = f'{name}_{key}'
                    expected.add(member)
                    value = torch.from_numpy(archive[member])
                    if value.dtype != torch.float32 or value.shape != p.shape or not torch.isfinite(value).all():
                        raise ValueError(f'Invalid {member} head at task {t}')
                    fitted[key] = value
                models[t][name] = fitted
            if set(archive.files) != expected:
                raise ValueError('Unexpected sealed parameters')
    return lock, models


def train_data(out, source, task, prior, roles):
    path = out / 'features' / f'task{task}' / 'train.npz'
    meta = read(path.with_suffix('.json'))
    cohort_path = source / 'cohorts' / f'task{task}.json'
    checkpoint_hash = sha256_file(source / 'compact_checkpoints' / f'task{task}.pth')
    cfg_hash = sha256_file(source / 'config.json')
    expected = dict(schema=1, task=task, split='train', smoke=False, batch_size=64,
                    reference_sha256=prior['train_feature_sha256'],
                    checkpoint_sha256=checkpoint_hash, source_config_sha256=cfg_hash)
    if (any(meta['protocol'].get(k) != v for k, v in expected.items())
            or not meta['exact_replay'] or not meta['visibility_mask_applied']
            or meta['features_sha256'] != sha256_file(path)
            or prior['checkpoint_sha256'] != checkpoint_hash
            or prior['cohort_sha256'] != sha256_file(cohort_path)):
        raise ValueError(f'Incorrect TRAIN feature cache provenance at task {task}')
    with np.load(path, allow_pickle=False) as a:
        if list(a['head_ids']) != [task] or tuple(a['view_order']) != VIEWS:
            raise ValueError('Incorrect TRAIN task or view order')
        d = {k: a[k] for k in ('reliable', 'targets', 'branch_logits', 'sample_ids')}
        cls = torch.from_numpy(a['head_cls'])
    valid = d['reliable']
    ids = d['sample_ids']
    n, classes = len(ids), TASK_SIZES[task]
    if (n != meta['people'] or len(set(ids)) != n or valid.dtype != bool or valid.shape != (n,)
            or d['targets'].shape != (n, classes) or not np.isin(d['targets'], (0, 1)).all()
            or not d['targets'].any(1).all() or d['branch_logits'].shape != (n, classes, 3)
            or cls.ndim != 4 or cls.shape[:3] != (n, 1, 3) or not cls.shape[-1]
            or meta['sample_id_sha256'] != hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()):
        raise ValueError(f'Invalid TRAIN arrays at task {task}')
    d['branch_logits'][~valid, :, 2] = 0
    if not np.isfinite(d['branch_logits']).all() or not torch.isfinite(cls).all() or cls[~valid, :, 2].count_nonzero():
        raise ValueError('Nonfinite active input or unmasked invalid Face')
    check_cohort(d, meta, read(cohort_path)['fusion'], roles, task)
    return d, cls[:, 0].float(), meta


def run(source, prior_a, out):
    torch.set_num_threads(1)
    cfg_path = source / 'config.json'
    config = read(cfg_path)
    prev = read(prior_a / 'centers_lock.json')
    roles_path = source / 'train_image_roles.json'
    roles = read(roles_path)
    lock = dict(protocol=PROTOCOL, source_config_sha256=sha256_file(cfg_path),
                train_roles_sha256=sha256_file(roles_path),
                prior_a_sha256=sha256_file(prior_a / 'centers_lock.json'), tasks={})
    if (config['mode'] != 'sample_holdout' or not config['protect_branches'] or config['smoke']
            or config['seed'] != 1 or config['tasks'] != 8
            or prev['source_config_sha256'] != lock['source_config_sha256']
            or prev['source_roles_sha256'] != lock['train_roles_sha256']):
        raise ValueError('Expected historical protected 90/10 branches and matching prior A run')
    lock_path = out / 'caran_fit_lock.json'
    if lock_path.is_file():
        old, _ = load_models(out)
        for k in ('source_config_sha256','train_roles_sha256','prior_a_sha256'):
            if old[k] != lock[k]:
                raise ValueError('Sealed training provenance differs')
        for task in PROTOCOL['tasks']:
            file = out / 'features' / f'task{task}' / 'train.npz'
            if sha256_file(file) != old['tasks'][str(task)]['train_features_sha256']:
                raise ValueError(f'Task {task} TRAIN cache changed after fitting')
        return
    for task in PROTOCOL['tasks']:
        prior = prev['tasks'][str(task)]
        d, cls, meta = train_data(out, source, task, prior, roles)
        v = torch.from_numpy(d['reliable'])
        z = torch.from_numpy(d['branch_logits']).float()
        y = torch.from_numpy(d['targets']).float()
        params, metrics = {}, {}
        for name in METHODS:
            model, metrics[name] = train_head(name, d, cls, v, y, z)
            if name == 'A' and model['bias'][:, 0].tolist() != prior['center']:
                raise ValueError(f'Task {task} failed exact replay of A')
            params.update({name + '_' + k: p.numpy() for k,p in model.items()})
        path = model_path(out, task)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **params)
        tmp.replace(path)
        row = dict(task=task, people=len(v), reliable_face=int(v.sum()), head_cls_dim=cls.shape[-1],
                   checkpoint_sha256=prior['checkpoint_sha256'], cohort_sha256=prior['cohort_sha256'],
                   train_features_sha256=meta['features_sha256'], model_sha256=sha256_file(path),
                   prior_A_replay=True, diagnostics=metrics)
        lock['tasks'][str(task)] = row
        print(f'task{task} fit {len(v)} TRAIN people; end BCE',
              {k: round(m['final_bce'], 6) for k,m in metrics.items()}, flush=True)
    write_json(lock_path, lock)
    load_models(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('source','prior-a','out'):
        parser.add_argument('--' + k, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.prior_a, args.out)
