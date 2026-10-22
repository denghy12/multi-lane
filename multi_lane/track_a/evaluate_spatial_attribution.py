"""Read-only on/off and same-image box controls, preserving the recipient views."""
import argparse
from .spatial_resume import checkpoint_source_git
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .cross_view_query_pairs import plan_for_source
from .paired_transforms import PairedFullPersonTransform, move_model_inputs
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import build_model, _torch_load, _sha256, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .runner import (compute_metrics, compact_model_state_dict, tensor_state_digest, git_metadata,
                     seen_indices, set_seed, summarize_tasks)


class SpatialAttributionDataset(Dataset):
    def __init__(self, source, plan, classes, swap=False):
        self.source, self.plan, self.classes, self.swap = source, plan, list(classes), swap
        sizes = {}
        self.boxes = []
        for path, bbox in zip(source.file_paths, source.body_bboxes):
            if path not in sizes:
                with Image.open(path) as image:
                    sizes[path] = image.size
            width, height = sizes[path]
            box = PairedFullPersonTransform.target_box(bbox, width, height)
            rw, rh = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
            scaled = None if box is None else (box[0]*rw/width, box[1]*rh/height, box[2]*rw/width, box[3]*rh/height)
            self.boxes.append(PairedFullPersonTransform.untruncated_geometry(
                scaled, (round((rh-224)/2), round((rw-224)/2), 224, 224), False))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        images, y = self.source[index]
        row = self.plan['rows'][index]
        if not torch.allclose(images['spatial_bbox'], self.boxes[index], atol=1e-6, rtol=0):
            raise RuntimeError('Analytical target coordinates disagree with actual Full crop')
        if not torch.allclose(images['bbox'], torch.tensor(row['geometry']), atol=1e-6, rtol=0):
            raise RuntimeError('Eligibility geometry disagrees with actual Full crop')
        if self.swap and row['eligible']:
            # Only position/shape changes. Keep recipient visibility and pixels.
            images = dict(images, spatial_bbox=self.boxes[row['donor_index']])
        return images, y[self.classes], self.source.sample_ids[index], index


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.source_run / 'config.json').read_text())
    summary = json.loads((args.source_run / 'seed_summary.json').read_text())
    assert summary['status'] == 'complete' and summary['config'] == config
    assert config['spatial_summary_mode'] == 'target' and config['git']['dirty'] is False
    assert all(0 <= task < config['max_tasks'] for task in args.checkpoint_tasks)
    meta = git_metadata(Path(__file__).resolve().parents[2]); assert not meta['dirty']
    set_seed(0, config['tf32']); device = torch.device('cuda')
    source = make_source(config, args.data_root, args.face_manifest_root, 'val')
    plan = plan_for_source(source, .9)
    model = build_model(config, load_openai_clip_visual(args.clip_checkpoint)).float().to(device)
    args.output.mkdir(parents=True)
    (args.output / 'pairing.json').write_text(json.dumps(plan, indent=2) + '\n')
    rows = {mode: [] for mode in ('on', 'off', 'box_swap')}
    detail, hashes = {}, {}
    frozen = tensor_state_digest(model.visual_encoder.state_dict())
    for task in args.checkpoint_tasks:
        path = args.source_run / 'compact_checkpoints' / f'task{task}.pth'
        hashes[str(path)] = _sha256(path)
        restore_compact_model_state(model, _torch_load(path), task, checkpoint_source_git(config, task))
        model.requires_grad_(False).eval()
        before = tensor_state_digest(compact_model_state_dict(model))
        reference = None
        detail[str(task)] = {}
        for mode in rows:
            model.spatial_summary_runtime_enabled = mode != 'off'
            model.spatial_summary_bank.reset_statistics()
            loader = DataLoader(SpatialAttributionDataset(source, plan, seen_indices(task), mode == 'box_swap'),
                                batch_size=64, shuffle=False, num_workers=0)
            ids, targets, indexes = [], [], []
            logits = {view: [] for view in ('fused', 'full', 'person', 'face')}
            with torch.no_grad():
                for batch, (images, y, sample_ids, index) in enumerate(loader):
                    if args.smoke_batches and batch >= args.smoke_batches:
                        break
                    with torch.cuda.amp.autocast(enabled=config['amp']):
                        fused, views = model.seen_logits_with_views(move_model_inputs(images, device))
                    for key, value in dict(fused=fused, **views).items():
                        logits[key].append(value.float().cpu())
                    ids.extend(sample_ids); targets.append(y); indexes.extend(index.tolist())
            tensors = {key: torch.cat(value) for key, value in logits.items()}
            y = torch.cat(targets)
            eligible = np.asarray([plan['rows'][i]['eligible'] for i in indexes], dtype=bool)
            probabilities = {key: value.sigmoid() for key, value in tensors.items()}
            metrics = {key: asdict(compute_metrics(task, value, y, .5)) for key, value in probabilities.items()}
            if reference is None:
                reference = tensors
                # Replay the separately archived normal full-cohort evaluation.
                folder = args.source_run.parent / 'full_val'
                with np.load(folder / 'scores' / f'task{task}.npz') as archive:
                    assert np.array_equal(archive['sample_ids'][:len(ids)], np.asarray(ids))
                    assert np.array_equal(archive['logits'][:len(ids)], tensors['fused'].numpy()), 'Normal replay failed'
            else:
                assert all(torch.equal(tensors[v], reference[v]) for v in ('person', 'face'))
                if mode == 'box_swap':
                    assert all(torch.equal(tensors[v][~eligible], reference[v][~eligible]) for v in tensors)
            subsets = {}
            for name, mask in [('eligible', eligible), ('ineligible', ~eligible)]:
                subsets[name] = dict(samples=int(mask.sum()), metrics={
                    view: asdict(compute_metrics(task, value[mask], y[mask], .5)) for view, value in probabilities.items()
                } if mask.any() else {})
            out = args.output / mode; out.mkdir(exist_ok=True)
            np.savez_compressed(out / f'task{task}.npz', sample_ids=np.asarray(ids), targets=y.numpy(),
                                eligible=eligible, **{k+'_logits': v.numpy() for k, v in tensors.items()})
            detail[str(task)][mode] = dict(metrics=metrics, subsets=subsets,
                                           spatial_statistics=model.spatial_summary_bank.statistics())
            rows[mode].append(metrics['fused'])
            assert tensor_state_digest(compact_model_state_dict(model)) == before
            print(f'task={task} condition={mode} mAP={metrics["fused"]["mAP"]:.6f}', flush=True)
        assert _sha256(path) == hashes[str(path)]
    assert tensor_state_digest(model.visual_encoder.state_dict()) == frozen
    result = dict(status='complete', split='val', checkpoint_tasks=args.checkpoint_tasks,
                  training_performed=False, source_unchanged=True, fixed_person_face=True,
                  noneligible_swap_unchanged=True, normal_archive_replay=True,
                  evaluator_git=meta, source_git=config['git'], source_sha256=hashes,
                  task_metrics=rows, details=detail, eligible_people=plan['eligible_people'],
                  eligible_images=plan['eligible_images'], full_cohort=not bool(args.smoke_batches),
                  test_used_for_selection=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'output', 'data-root', 'clip-checkpoint', 'face-manifest-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint-tasks', type=int, nargs='+', default=[0, 2, 7])
    parser.add_argument('--smoke-batches', type=int, default=0)
    evaluate(parser.parse_args())
