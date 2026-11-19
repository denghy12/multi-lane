"""Recompute TRAIN features after reducing only one auxiliary crop's resolution."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.runner import set_seed
from scripts.emotic.d9_disagreement_source import load_source
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.extract_d9_task10_probe import read_json, train_loader
from scripts.emotic.fit_d9_task10_probe import PROJECTION_SEED, project
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def degrade_view(images, view):
    """Update both the recipient crop and its sender copy without modifying original inputs."""
    if view not in ('person', 'face'):
        raise ValueError('Only Person or reliable Face can be degraded')
    output = images.copy()
    for name in (view, 'evidence_' + view):
        x = images[name]
        if x.ndim != 4 or x.shape[-2:] != (224, 224):
            raise ValueError('Expected a batch of normalized 224x224 crops')
        reduced = F.interpolate(x.float(), size=(112, 112), mode='area')
        up = F.interpolate(reduced, size=(224, 224), mode='bilinear', align_corners=False).to(x.dtype)
        if view == 'face':
            up = torch.where(images['evidence_face_reliable'].bool().reshape(-1, 1, 1, 1), up, x)
        output[name] = up
    return output


def forward(model, images):
    capture = {}
    with torch.autocast('cuda'):
        native = model(images, all_seen=False, task10_probe=capture)
    r = images['evidence_face_reliable'].reshape(-1).bool()
    relation = capture['relation'].float().clone()
    relation[~r, 4:] = 0
    content = torch.stack([capture[v]['content'].float() for v in VIEWS], dim=1)
    content[~r, 2] = 0
    delta = torch.stack([capture[v]['delta'] for v in ('person', 'face')], dim=1)
    delta[~r, 1] = 0
    logits = torch.stack([native['branches'][v] for v in VIEWS], dim=-1)
    logits[~r, :, 2] = 0
    return {k: a.cpu().numpy() for k, a in
            dict(branch_logits=logits, content=content, relation=relation, delta=delta).items()}


def verify_replay(observed, reference, rows):
    for k, a in observed.items():
        np.testing.assert_array_equal(a, reference[k][rows], err_msg=f'Natural TRAIN replay: {k}')


@torch.inference_mode()
def collect(model, loader, natural, views, smoke):
    chunks = {v: {k: [] for k in ('branch_logits', 'content_proj', 'relation')} for v in views}
    offset = 0
    for images, targets, ids in loader:
        rows = slice(offset, offset + len(ids))
        inputs = move_model_inputs(images, torch.device('cuda'))
        reliable = natural['reliable'][rows]
        actual = forward(model, inputs)
        actual.update(sample_ids=np.asarray(ids), targets=targets.numpy(),
                      reliable=inputs['evidence_face_reliable'].reshape(-1).cpu().numpy())
        verify_replay(actual, natural, rows)
        for view in views:
            altered = forward(model, degrade_view(inputs, view))
            np.testing.assert_array_equal(altered['branch_logits'][:, :, 0], actual['branch_logits'][:, :, 0])
            np.testing.assert_array_equal(altered['content'][:, 0], actual['content'][:, 0])
            if view == 'face':
                for k in altered:
                    np.testing.assert_array_equal(altered[k][~reliable], actual[k][~reliable])
            altered['content_proj'] = project(altered)['content']
            for k, target in chunks[view].items():
                if not np.isfinite(altered[k]).all():
                    raise ValueError(f'Nonfinite {view} variant {k}')
                target.append(altered[k])
        offset = rows.stop
        if smoke:
            break
    if offset != (min(64, len(natural['sample_ids'])) if smoke else len(natural['sample_ids'])):
        raise ValueError('Wrong number of emitted TRAIN people')
    for view, by_field in chunks.items():
        arrays = {k: np.concatenate(values) for k, values in by_field.items()}
        arrays.update({k: natural[k][:offset] for k in ('sample_ids', 'targets', 'reliable')})
        yield view, arrays


def extract(source, cache, cache6, out, tasks, workers, smoke):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    train = build_joint_source(input_paths(), 'train')
    roles = global_image_roles(train, 'sample_full', 1)
    if roles != read_json(source / 'train_image_roles.json'):
        raise ValueError('Original TRAIN image roles do not replay')
    train = fusion_source(train)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        natural, ident = load_source(cache6 if task == 6 else cache, source, task, 'train')
        spec = dict(task=task, split='train', smoke=smoke, checkpoint_sha256=ident['checkpoint_sha256'],
                    natural_sha256=ident['sha256'], config_sha256=ident['config_sha256'], schema=1,
                    input_size=224, reduced_size=112, projection_seed=PROJECTION_SEED, batch_size=64)
        views = []
        for view in ('person', 'face'):
            p = out / f'task{task}_{view}.npz'
            manifest = p.with_suffix('.json')
            if manifest.is_file():
                meta = read_json(manifest)
                if not all(meta.get(k) == v for k, v in dict(spec, view=view).items()):
                    raise ValueError(f'Existing protocol differs: {manifest}')
                if p.is_file() and sha256_file(p) == meta['archive_sha256']:
                    continue
            views.append(view)
        if not views:
            print(f'task{task}: existing pair hashes verified', flush=True)
            continue
        model = load_evaluation_model(source, task).requires_grad_(False).eval()
        loader = train_loader(source, train, roles, task, workers, 64)
        for view, data in collect(model, loader, natural, views, smoke):
            path = out / f'task{task}_{view}.npz'
            temp = path.with_suffix(f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, **data)
            temp.replace(path)
            write_json(path.with_suffix('.json'), dict(spec, view=view, people=len(data['sample_ids']),
                       original_replayed=True, archive_sha256=sha256_file(path)))
            print(f'task{task} {view}: replay verified, wrote {len(data["sample_ids"])} TRAIN people', flush=True)
        del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'cache', 'cache6', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', type=int, choices=(0, 3, 6), default=[0, 3, 6])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    extract(**vars(parser.parse_args()))
