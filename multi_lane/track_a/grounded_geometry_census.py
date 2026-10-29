"""TRAIN-only one-augmentation-pass census; never changes the main training stream."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .person_grounded_summary import patch_coverage
from .protected_joint_experiment import runtime, training_source, single_thread_worker
from .runner import dataset_view, task_indices, set_seed
from .run_historical_b32_test_attribution import read, write
from .export_compact_test_scores import _sha256


def main(args):
    git = runtime(); set_seed(1, True)
    base = read(args.source_run/'config.json')
    source = training_source(base, args); records = {}
    for task in range(8):
        view = dataset_view(source, task_indices(task), True)
        loader = DataLoader(view, batch_size=64, shuffle=True, num_workers=2, prefetch_factor=1,
            multiprocessing_context='spawn', worker_init_fn=single_thread_worker,
            generator=torch.Generator().manual_seed(930650+task))
        total = eligible = valid = updates = eligible_updates = reliable = 0; images = set()
        for x, _, ids in loader:
            rho, ok, e = patch_coverage(x['spatial_bbox'], x['spatial_visibility'], 196)
            total += len(ids); valid += int(ok.sum()); eligible += int(e.sum())
            reliable += int(x['face_reliable'].sum()); updates += 1; eligible_updates += int(e.any())
            images.update(str(s).rsplit('#person=', 1)[0] for s in ids)
        assert total == len(view)
        records[str(task)] = dict(people=total, original_images=len(images), any_visible=valid,
            eligible_loc=eligible, eligible_fraction=eligible/total, reliable_face=reliable,
            batches=updates, batches_with_loc=eligible_updates)
        print('TRAIN_GEOMETRY_CENSUS', task, records[str(task)], flush=True)
    if not all(v['eligible_loc'] > 0 for v in records.values()):
        raise ValueError('A task has no valid localization supervision; do not launch grounded training')
    write(args.output/'result.json', dict(status='complete', tasks=list(range(8)), source_git=git,
        split='train', evaluation_data_read=False, model_loaded=False, training_updates=0,
        transform='historical_train_augmentation_one_fixed_pass', sampler_seed='930650+task',
        min_visibility=.9, source_configuration_sha256=_sha256(args.source_run/'config.json'), records=records))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'data-root', 'face-manifest-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
