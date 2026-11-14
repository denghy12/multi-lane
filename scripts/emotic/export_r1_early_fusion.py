"""Export task-private epoch-15 TRAIN predictions from a completed population run."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a import runner
from multi_lane.track_a.fusion_population import population_indices
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import (
    model_from_config, read, sha, single_thread_worker, write,
)
from scripts.emotic.r1_asymmetric_forward import AsymmetricReceivers
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source
from scripts.emotic.run_r1_single_trajectory_fusion import VIEWS, branch_paths, save_npz

BATCH = 64
EPOCH = 15
REFERENCE_EPOCH = 30


def restore(models, source_root, task, epoch):
    path = source_root / 'recovery' / f'task{task}' / f'epoch{epoch:02d}.pth'
    digest = sha(path)
    state = torch.load(path, map_location='cpu')
    if (state['format'] != 'single_trajectory_train_population_v1'
            or state['condition'] != 'holdout_return' or state['fit_split'] != 'train'
            or (state['task'], state['epoch']) != (task, epoch)):
        raise ValueError(f'Unexpected whole checkpoint: {path}')
    updates = {}
    for view, model in models.items():
        branch = state['branches'][view]
        training = branch['training']
        if (branch['task'] != task or training['next_epoch'] != epoch
                or len(training['history']) != epoch
                or any(row['skipped_optimizer_steps'] for row in training['history'])):
            raise ValueError(f'Unexpected {view} training boundary: {path}')
        keys = model.load_state_dict(branch['model'], strict=False)
        if keys.unexpected_keys or any(not k.startswith('visual_encoder.') for k in keys.missing_keys):
            raise ValueError(f'Incomplete {view} model: {path}')
        model.restore_task(task)
        model.eval()
        updates[view] = training['completed_task_updates']
    return path, digest, updates


def load_reference(root, task, ids):
    reference = {}
    for view in VIEWS:
        path = root/'population_observations'/f'task{task}'/f'{view}.npz'
        with np.load(path, allow_pickle=False) as d:
            row = {k: d[k] for k in ('ids', 'targets', 'reliable', 'logits')}
        if row['ids'].tolist() != ids:
            raise ValueError(f'{view}: historical TRAIN person list changed')
        if reference and any(not np.array_equal(row[k], reference['full'][k])
                             for k in ('targets', 'reliable')):
            raise ValueError(f'{view}: historical TRAIN labels or masks changed')
        reference[view] = row
    return reference


@torch.no_grad()
def observe(models, source, task, groups, workers, spot):
    """Decode each batch once, retaining the two independent source batch boundaries."""
    classes = list(runner.task_indices(task))
    logits = {v: [] for v in VIEWS}
    targets, reliable, ids = [], [], []
    for group, indices in groups.items():
        if spot:
            indices = indices[:BATCH]
        if not indices:
            continue
        data = runner.LabelView(source, indices, classes, include_sample_id=True)
        options = dict(prefetch_factor=1, worker_init_fn=single_thread_worker) if workers else {}
        loader = DataLoader(data, batch_size=BATCH, num_workers=workers, **options,
                            generator=torch.Generator().manual_seed(77191+task))
        for images, labels, names in loader:
            if not torch.all(labels.any(dim=1) == (group == 'current_positive')):
                raise ValueError('TRAIN observation violates population membership')
            images = move_model_inputs(images, torch.device('cuda'))
            ids.extend(names)
            targets.append(labels.numpy().astype(np.float32))
            reliable.append(images['evidence_face_reliable'].bool().cpu().numpy())
            for view, model in models.items():
                with torch.autocast('cuda'):
                    batch = model.seen_logits(AsymmetricReceivers.receiver_inputs(images, view))[:, classes]
                logits[view].append(batch.float().cpu().numpy())
    common = dict(ids=np.asarray(ids, dtype=str), targets=np.concatenate(targets),
                  reliable=np.concatenate(reliable))
    rows = {view: dict(**common, logits=np.concatenate(values)) for view, values in logits.items()}
    if any(not np.isfinite(r['logits']).all() for r in rows.values()):
        raise FloatingPointError('Nonfinite TRAIN prediction')
    return rows


def check_rows(rows, reference, positions, check_logits):
    fields = ('ids', 'targets', 'reliable', 'logits') if check_logits else ('ids', 'targets', 'reliable')
    for view, row in rows.items():
        for key in fields:
            expected = reference[view][key][positions]
            if not np.array_equal(row[key], expected):
                detail = (f' max_abs_error={np.max(np.abs(row[key]-expected))}'
                          if key == 'logits' and row[key].shape == expected.shape else '')
                raise ValueError(f'{view}/{key} does not replay the TRAIN reference.{detail}')


def export_task(models, source, source_root, root, task, workers, replay_only):
    start = time.monotonic()
    selection = population_indices(source, task)
    groups = {k: selection[k] for k in ('current_positive', 'added_negative')}
    indices = groups['current_positive'] + groups['added_negative']
    ids = [str(source.sample_ids[i]) for i in indices]
    if len(set(ids)) != len(ids) or not all(i.startswith('train:') for i in ids):
        raise ValueError('Expected unique official TRAIN people')
    ref = load_reference(source_root, task, ids)
    pos = len(groups['current_positive'])
    slices = list(range(min(pos, BATCH))) + list(range(pos, min(pos+BATCH, len(indices))))
    audit = dict(task=task, source_root=str(source_root), fit_split='train',
                 source_images_per_pass=len(indices), current_positive=pos, added_negative=len(indices)-pos,
                 epoch=EPOCH, batch_size=BATCH, workers=workers, torch_threads=torch.get_num_threads(),
                 replay_people=len(slices), device=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 source_checkpoints={}, source_update_counts={}, output_hashes={})
    # Replay the actual e30 archived observations before using the identical path for e15.
    for epoch in ([REFERENCE_EPOCH] if replay_only else [REFERENCE_EPOCH, EPOCH]):
        spot = epoch == REFERENCE_EPOCH
        path, before, updates = restore(models, source_root, task, epoch)
        readout = observe(models, source, task, groups, workers, spot)
        check_rows(readout, ref, slices if spot else slice(None), check_logits=spot)
        if before != sha(path):
            raise RuntimeError(f'Checkpoint changed while exporting: {path}')
        audit['source_checkpoints'][str(epoch)] = before
        audit['source_update_counts'][str(epoch)] = updates
        if not spot:
            for view, arrays in readout.items():
                target = root/'observations'/f'task{task}'/f'{view}.npz'
                save_npz(target, **arrays)
                audit['output_hashes'][view] = sha(target)
    batches = sum(math.ceil(len(ix)/BATCH) for ix in groups.values()) if not replay_only else 0
    audit.update(status='complete', source_checkpoint_sha_unchanged=True,
                 epoch30_replay_exact=True, branch_training_updates=0,
                 epoch15_decode_batches=batches, epoch15_view_forward_batches=batches*len(VIEWS),
                 seconds=round(time.monotonic()-start, 3))
    return audit


def run(args):
    if args.workers < 0 or not args.tasks or len(set(args.tasks)) != len(args.tasks):
        raise ValueError('Expected a nonempty shard of distinct tasks and nonnegative workers')
    source_root, root = args.source, args.root
    if root.resolve() == source_root.resolve() or source_root.resolve() in root.resolve().parents:
        raise ValueError('Choose a separate output from the source run')
    protocol = read(source_root/'protocol.json')
    if (protocol['seed'] != 1 or protocol['smoke'] or protocol['condition'] != 'holdout_return'
            or not (source_root/'completion.json').is_file()):
        raise ValueError('Expected completed formal seed1 holdout-return population run')
    torch.set_num_threads(2)
    runner.set_seed(1, True)
    inputs = input_paths()
    paths = branch_paths(source_root)
    source = load_source(paths['person'], inputs, 'train')
    models = {v: model_from_config(read(paths[v]/'config.json'), inputs.clip_checkpoint) for v in VIEWS}
    folder = root/('replay_audit' if args.replay_only else 'export_audit')
    folder.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        marker = folder/f'task{task}.json'
        with (folder/f'task{task}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if marker.exists():
                record = read(marker)
                if (record['status'] != 'complete' or record['source_root'] != str(source_root)
                        or any(sha(source_root/'recovery'/f'task{task}'/f'epoch{int(e):02d}.pth') != digest
                               for e, digest in record['source_checkpoints'].items())
                        or any(sha(root/'observations'/f'task{task}'/f'{v}.npz') != digest
                               for v, digest in record['output_hashes'].items())):
                    raise ValueError(f'Existing export differs: {marker}')
                print('Reuse verified', marker, flush=True)
                continue
            record = export_task(models, source, source_root, root, task, args.workers, args.replay_only)
            write(marker, record)
            print('Verified', marker, record['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--tasks', nargs='+', type=int, choices=range(8), required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--replay-only', action='store_true')
    run(parser.parse_args())
