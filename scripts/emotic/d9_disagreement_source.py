"""Read current-head features only from the verified, protected full-TRAIN branch trajectory."""
import hashlib
from pathlib import Path

import numpy as np

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import task_indices
from scripts.emotic.d9_sample_training import image_key
from scripts.emotic.d9_tcp_probe import identify_source, read
from scripts.emotic.fit_d9_task10_probe import load_arrays


def _check_train(data, classes, id_hash, cohort, roles):
    y, valid = data['targets'], data['reliable']
    images = {image_key(person) for person in data['sample_ids']}
    measured = dict(people=len(y), images=len(images), classes=list(classes),
                    sample_id_sha256=id_hash, reliable_face=int(valid.sum()),
                    positives=y.sum(0).astype(int).tolist(),
                    reliable_face_positives=y[valid].sum(0).astype(int).tolist())
    if (cohort['branch'] != cohort['fusion'] or any(cohort['fusion'][k] != v for k, v in measured.items())
            or not y.any(axis=1).all() or any(roles.get(i) != 'both' for i in images)):
        raise ValueError('Cache does not replay all eligible people in the current TRAIN cohort')


def load_source(root, modelroot, task, split):
    """Return arrays plus JSON-compatible identity for use in the Validation seal."""
    if task not in (0, 3, 6) or split not in ('train', 'val', 'test'):
        raise ValueError('Expected a registered task/split')
    root, modelroot = Path(root), Path(modelroot)
    path, identity = identify_source(root, task, split, source_mode='sample_full')
    meta = read(path.with_suffix('.json'))
    protocol = meta['protocol']
    cfg = modelroot / 'config.json'
    cfg_data = read(cfg)
    required = dict(mode='sample_full', smoke=False, seed=1, tasks=8, epochs=30,
                    batch_size=64, image_split='all_train', protect_branches=True)
    cp = modelroot/'compact_checkpoints'/f'task{task}.pth'
    if (any(cfg_data.get(k) != v for k, v in required.items())
            or sha256_file(cfg) != identity['config_sha256']
            or sha256_file(cp) != identity['checkpoint_sha256']
            or protocol['batch_size'] != 64 or protocol['schema'] != 2):
        raise ValueError(f'Cache does not belong to the full TRAIN task{task} checkpoint')

    classes = task_indices(task)
    data = load_arrays(path, task)
    ids = data['sample_ids']
    id_hash = hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
    if (id_hash != meta['sample_id_sha256'] or len(ids) != meta['people'] or
            int(data['reliable'].sum()) != meta['reliable_face']):
        raise ValueError(f'Cache metadata does not match its people for task{task}/{split}')

    if split == 'train':
        reference = modelroot/'cohorts'/f'task{task}.json'
        _check_train(data, classes, id_hash, read(reference), read(modelroot/'train_image_roles.json'))
    else:
        reference = modelroot/'evaluation'/split/f'task{task}.npz'
    ref_digest = sha256_file(reference)
    if ref_digest != protocol['source_cohort_or_archive_sha256']:
        raise ValueError(f'Changed source {reference}')

    if split != 'train':
        ref_meta = read(reference.with_suffix('.json'))
        ref_spec = ref_meta['protocol']
        expected = dict(split=split, task=task, smoke=False, batch_size=64,
                        checkpoint_sha256=identity['checkpoint_sha256'], config_sha256=identity['config_sha256'])
        if any(ref_spec[k] != v for k, v in expected.items()) or ref_meta['scores_sha256'] != ref_digest:
            raise ValueError(f'Original predictions lack the expected checkpoint / split identity: {reference}')
        # load_arrays zeroes unavailable Face outputs for downstream use. Verify the raw caches
        # before this transformation, including the unavailable branch's original scores.
        with np.load(reference, allow_pickle=False) as native, np.load(path, allow_pickle=False) as cached:
            if native['targets'].shape != (len(ids), classes[-1] + 1):
                raise ValueError('Unexpected native class axis for the introduction stage')
            for key in ('sample_ids', 'reliable', 'targets', 'branch_logits', 'fixed_logits', 'equal_logits'):
                source = native[key]
                if key not in ('sample_ids', 'reliable'):
                    source = source[:, classes]
                if not np.array_equal(cached[key], source):
                    raise ValueError(f'Current-head {key} diverges from the native forward at {reference}')
    identity.update(task=task, split=split, source_mode='sample_full', reference_sha256=ref_digest,
                    sample_id_sha256=id_hash, people=len(ids), reliable_face=int(data['reliable'].sum()),
                    class_indices=list(classes))
    return data, identity
