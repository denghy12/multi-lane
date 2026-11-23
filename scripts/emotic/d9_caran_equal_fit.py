"""Refit the four historical CARAN readouts on the 13,950-update 90% trajectory."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import check_cohort, group_proportions
from scripts.emotic.d9_caran_eight_fit import (
    METHODS, VIEWS, PROTOCOL as HISTORICAL_PROTOCOL, initialize, model_path,
    readout_weights, train_head,
)
from scripts.emotic.d9_matched_natural_fit import fit_natural
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.extract_d9_caran_equal import audit_source, reference

PROTOCOL = dict(HISTORICAL_PROTOCOL, version=2,
                source='protected 90% original-image split, 13950 branch updates, zero skipped',
                fitting='natural held TRAIN only; no blur variants, no Validation/Test gradients',
                comparison='all four historical readout formulas and optimizer settings unchanged')


def train_data(out, source, task, audit, roles):
    path = out/'features'/f'task{task}'/'train.npz'
    meta = read(path.with_suffix('.json'))
    cohort_path = source/'cohorts'/f'task{task}.json'
    cp_hash = sha256_file(source/'compact_checkpoints'/f'task{task}.pth')
    ref_path = reference(source, task, 'train', audit['source_config_sha256'], cp_hash)
    expected = dict(schema=1, task=task, split='train', smoke=False, batch_size=64,
                    reference_sha256=sha256_file(ref_path), checkpoint_sha256=cp_hash,
                    source_config_sha256=audit['source_config_sha256'])
    if (meta['protocol'] != expected or not meta['exact_replay']
            or not meta['visibility_mask_applied'] or meta['source_update_audit'] != audit):
        raise ValueError(f'Unexpected TRAIN feature provenance: {path}')
    require_digest(path, meta['features_sha256'])
    with np.load(path, allow_pickle=False) as a, np.load(ref_path, allow_pickle=False) as ref:
        if list(a['head_ids']) != [task] or tuple(a['view_order']) != VIEWS:
            raise ValueError('Incorrect TRAIN task or view axes')
        d = {k: a[k] for k in ('reliable', 'targets', 'branch_logits', 'sample_ids')}
        for key, value in d.items():
            np.testing.assert_array_equal(value, ref[key], err_msg=f'TRAIN reference mismatch: {key}')
        cls = torch.from_numpy(a['head_cls'])
    ids, valid = d['sample_ids'], d['reliable']
    n, classes = len(ids), TASK_SIZES[task]
    if (n != meta['people'] or len(set(ids)) != n or valid.dtype != bool or valid.shape != (n,)
            or d['targets'].shape != (n, classes) or not np.isin(d['targets'], (0, 1)).all()
            or not d['targets'].any(1).all() or d['branch_logits'].shape != (n, classes, 3)
            or d['branch_logits'].dtype not in (np.float16, np.float32) or cls.shape != (n, 1, 3, 512)
            or meta['sample_id_sha256'] != hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()):
        raise ValueError(f'Invalid TRAIN arrays at task{task}')
    if (not np.isfinite(d['branch_logits']).all() or not torch.isfinite(cls).all()
            or cls[~valid, :, 2].count_nonzero() or np.count_nonzero(d['branch_logits'][~valid, :, 2])):
        raise ValueError('Nonfinite input or nonzero unavailable Face')
    check_cohort(d, meta, read(cohort_path)['fusion'], roles, task)
    return d, cls[:, 0].float(), meta, sha256_file(cohort_path)


def load_models(out):
    out = Path(out)
    lock = read(out/'caran_fit_lock.json')
    if lock['protocol'] != PROTOCOL or set(lock['tasks']) != {str(t) for t in PROTOCOL['tasks']}:
        raise ValueError('Incomplete or incompatible equal-update TRAIN lock')
    models = {}
    for task in PROTOCOL['tasks']:
        row = lock['tasks'][str(task)]
        path = model_path(out, task)
        require_digest(path, row['model_sha256'])
        require_digest(out/'features'/f'task{task}'/'train.npz', row['train_features_sha256'])
        require_digest(out/'features'/f'task{task}'/'train.json', row['train_meta_sha256'])
        models[task] = {}
        with np.load(path, allow_pickle=False) as a:
            expected = set()
            for method in METHODS:
                template = initialize(method, TASK_SIZES[task], 3*row['head_cls_dim'])
                fitted = {}
                for key, example in template.items():
                    member = method+'_'+key
                    expected.add(member)
                    value = torch.from_numpy(a[member])
                    if value.dtype != torch.float32 or value.shape != example.shape or not value.isfinite().all():
                        raise ValueError(f'Invalid sealed parameters: task{task}/{member}')
                    fitted[key] = value
                models[task][method] = fitted
            if set(a.files) != expected:
                raise ValueError('Unexpected sealed parameters')
    return lock, models


def verify_source(source, lock):
    audit = audit_source(source)
    if audit != lock['source_update_audit']:
        raise ValueError('Source training history changed after fitting')
    for task in PROTOCOL['tasks']:
        row = lock['tasks'][str(task)]
        require_digest(source/'compact_checkpoints'/f'task{task}.pth', row['checkpoint_sha256'])
        require_digest(source/'cohorts'/f'task{task}.json', row['cohort_sha256'])
        require_digest(source/'paired_A'/f'task{task}_held.npz', row['reference_sha256'])
    return audit


def save_params(path, params):
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(params):
                raise ValueError(f'Incompatible existing fitted parameters: {path}')
            for key, value in params.items():
                np.testing.assert_array_equal(previous[key], value)
        return
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **params)
    tmp.replace(path)


def run(source, out, natural=None):
    torch.set_num_threads(1)
    audit = audit_source(source)
    roles = read(source/'train_image_roles.json')
    out.mkdir(parents=True, exist_ok=True)
    natural_path = natural/'centers_lock.json' if natural is not None else None
    natural_lock = read(natural_path) if natural_path is not None else None
    if natural_lock is not None:
        if (natural_lock['config_sha256'] != audit['source_config_sha256']
                or natural_lock['roles_sha256'] != audit['roles_sha256']):
            raise ValueError('Natural-only replay belongs to another branch trajectory')
    natural_hash = sha256_file(natural_path) if natural_path is not None else None
    lock_path = out/'caran_fit_lock.json'
    if lock_path.exists():
        old, _ = load_models(out)
        verify_source(source, old)
        if old['natural_reference_sha256'] != natural_hash:
            raise ValueError('Natural-only replay reference changed')
        print('Reusing all eight verified sealed TRAIN readouts', flush=True)
        return
    lock = dict(protocol=PROTOCOL, source_update_audit=audit,
                source_config_sha256=audit['source_config_sha256'],
                train_roles_sha256=audit['roles_sha256'],
                natural_reference_sha256=natural_hash, tasks={})
    for task in PROTOCOL['tasks']:
        d, cls, meta, cohort_hash = train_data(out, source, task, audit, roles)
        valid = torch.from_numpy(d['reliable'])
        targets = torch.from_numpy(d['targets']).float()
        logits = torch.from_numpy(d['branch_logits']).float()
        params, diagnostics = {}, {}
        for method in METHODS:
            model, diagnostics[method] = train_head(method, d, cls, valid, targets, logits)
            if method == 'A':
                measured = np.asarray(group_proportions(model['bias'][:, 0].tolist()))
                if natural_lock is not None:
                    row = natural_lock['tasks'][str(task)]
                    if (row['archive_sha256'] != meta['protocol']['reference_sha256']
                            or row['checkpoint_sha256'] != meta['protocol']['checkpoint_sha256']
                            or row['cohort_sha256'] != cohort_hash
                            or row['train_ids_sha256'] != meta['sample_id_sha256']):
                        raise ValueError(f'Natural-only TRAIN reference mismatch at task{task}')
                    expected = np.asarray(row['natural']['weights'])
                else:
                    # Independent reduction order, matching the existing natural-only protocol.
                    expected = np.asarray(fit_natural(d)['weights'])
                np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-5)
                diagnostics[method]['natural_replay_max_weight_difference'] = float(abs(measured-expected).max())
            params.update({method+'_'+key: value.numpy() for key, value in model.items()})
        path = model_path(out, task)
        save_params(path, params)
        lock['tasks'][str(task)] = dict(task=task, people=len(valid), reliable_face=int(valid.sum()),
            head_cls_dim=cls.shape[-1], checkpoint_sha256=meta['protocol']['checkpoint_sha256'],
            cohort_sha256=cohort_hash, reference_sha256=meta['protocol']['reference_sha256'],
            train_features_sha256=meta['features_sha256'],
            train_meta_sha256=sha256_file(out/'features'/f'task{task}'/'train.json'),
            model_sha256=sha256_file(path), diagnostics=diagnostics)
        print(f'task{task} fit {len(valid)} held TRAIN people:',
              {m: round(v['final_bce'], 6) for m, v in diagnostics.items()}, flush=True)
    write_verified(lock_path, lock)
    load_models(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--natural', type=Path, help='Optional matched natural-only A reference directory')
    run(**vars(p.parse_args()))
