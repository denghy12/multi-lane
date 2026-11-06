"""Evaluate each site switch with one input decode per batch."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import model_from_config, read, sha, single_thread_worker, write
from multi_lane.track_a.runner import dataset_view, seen_indices, set_seed, write_evaluation_scores
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, restore, score_suffix


TASKS = (0, 2, 7)
VARIANTS = ('early_only', 'late_only', 'both_off')


def evaluate_switches(train, output):
    inputs = input_paths()
    set_seed(1, True)
    torch.set_num_threads(2)
    model = model_from_config(read(train / 'config.json'), inputs.clip_checkpoint)
    early, late = model.frozen_cross_layers
    modes = dict(zip(VARIANTS, ((early,), (late,), ())))
    checkpoints = {train / 'compact_checkpoints' / f'task{task}.pth' for task in TASKS}
    hashes = {str(path): sha(path) for path in sorted(checkpoints)}
    device = torch.device('cuda')
    for split in ('val', 'test'):
        source = load_source(train, inputs, split)
        for task in TASKS:
            restore(model, train, task)
            model.requires_grad_(False).eval()
            data = dataset_view(source, seen_indices(task), include_sample_id=True)
            loader = DataLoader(data, batch_size=64, num_workers=2, prefetch_factor=1,
                                worker_init_fn=single_thread_worker,
                                generator=torch.Generator().manual_seed(77191 + task))
            logits = {name: [] for name in modes}
            labels, ids = [], []
            with torch.no_grad():
                for images, target, batch_ids in loader:
                    images = move_model_inputs(images, device)
                    labels.append(target.float().cpu())
                    ids.extend(batch_ids)
                    for name, active in modes.items():
                        model.r1_enabled = bool(active)
                        model.frozen_cross_active_layers = active
                        # Separate autocast contexts avoid leaking cached weights across switches.
                        with torch.autocast('cuda'):
                            row = model.seen_logits(images).float().cpu()
                        logits[name].append(row)
            targets = torch.cat(labels)
            for name, batches in logits.items():
                scores = torch.cat(batches)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(f'Nonfinite {name} logits')
                write_evaluation_scores(output / name / score_suffix(split, task), task, ids,
                                        scores, targets, probabilities=torch.cat([b.sigmoid() for b in batches]),
                                        batch_lengths=[len(b) for b in batches])
            print('SITE_SWITCHES', split, task, len(ids), flush=True)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError('Source checkpoint changed')
    write(output / 'completion.json', dict(status='complete', tasks=TASKS, checkpoint_hashes=hashes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate_switches(args.training_root, args.output)
