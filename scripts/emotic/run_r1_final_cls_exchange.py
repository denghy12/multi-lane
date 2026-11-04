"""Fit task-specific cross-view corrections using the final R1 CLS tokens."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.evaluation_scores import load_evaluation_scores
from multi_lane.track_a.fuse_fixed_three_view_test import (
    LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS,
)
from multi_lane.track_a.r1_crossvit_experiment import flags, model_from_config, read, write
from multi_lane.track_a.runner import (
    build_transforms, compute_metrics, dataset_view, seen_indices,
    set_seed, summarize_tasks, task_indices,
)

VIEWS = ('full', 'person', 'face')
KINDS = ('mlp', 'attention')
CAP = .05


class Exchange(nn.Module):
    def __init__(self, width, kind, dim=64):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(width, dim, bias=False)
        self.view_embedding = nn.Parameter(torch.zeros(3, dim))
        if kind == 'attention':
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        else:
            self.mix = nn.Linear(3 * dim, dim, bias=False)
        self.output = nn.Linear(dim, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, reliable):
        tokens = self.project(features) + self.view_embedding
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        valid[:, 2] = reliable
        tokens = tokens.masked_fill(~valid[..., None], 0)
        if self.kind == 'attention':
            q, k, v = self.qkv(tokens).chunk(3, dim=-1)
            scores = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
            other = ~torch.eye(3, dtype=torch.bool, device=tokens.device)
            scores = scores.masked_fill(~(other & valid[:, None, :]), float('-inf'))
            mixed = scores.softmax(-1) @ v
        else:
            mixed = F.gelu(self.mix(torch.cat(
                [tokens.roll(-offset, dims=1) for offset in range(3)], dim=-1)))
        delta = self.output(mixed)
        delta = delta * (CAP / delta.norm(dim=-1, keepdim=True).clamp_min(CAP))
        return delta


def dataset(config, split, inputs):
    _, transform = build_transforms(
        config['input_normalization'], config['train_crop_scale'], input_mode=config['input_mode'],
        person_transform_mode=config['person_transform_mode'],
        person_color_jitter_strength=config['person_color_jitter_strength'],
        person_color_jitter_probability=config['person_color_jitter_probability'],
    )
    return EMOTIC(str(inputs.data_root.parent), train=split == 'train',
                  eval_splits=(split,), transform=transform, input_mode=config['input_mode'],
                  person_crop_margin=config['person_crop_margin'],
                  face_manifest_root=inputs.face_manifest_root if config['input_mode'] == 'face_crop' else None)


@torch.no_grad()
def extract(models, sources, stage, split, smoke):
    features, logits = [], []
    ids, targets = None, None
    indices = task_indices(stage) if split == 'train' else seen_indices(stage)
    for model, source in zip(models, sources):
        view = dataset_view(source, indices, include_sample_id=True)
        if smoke:
            view = Subset(view, range(min(64, len(view))))
        loader = DataLoader(view, batch_size=64, num_workers=2, prefetch_factor=1, shuffle=False)
        batch_ids, vectors, scores, labels = [], [], [], []
        for images, y, names in loader:
            with torch.autocast('cuda'):
                h = model.encode_lanes(images.cuda(), all_seen_lanes=split != 'train')
                mask = model.task_class_mask[:stage+1] if split != 'train' else model.task_class_mask[stage:stage+1]
                z = (model.head(h) * mask.to(torch.float16)).sum(dim=1)[:, list(indices)]
            vectors.append(h.cpu())
            scores.append(z.float().cpu())
            labels.append(y.float())
            batch_ids.extend(names)
        y = torch.cat(labels)
        if ids is not None and (batch_ids != ids or not torch.equal(targets, y)):
            raise ValueError(f'Misaligned views at {split} stage {stage}')
        ids, targets = batch_ids, y
        features.append(torch.cat(vectors))
        logits.append(torch.cat(scores))
    face = sources[2]
    reliability = dict(zip(face.sample_ids, face.face_reliable))
    reliable = torch.tensor([reliability[i] for i in ids])
    weights = torch.where(reliable[:, None], torch.from_numpy(LOCKED_RELIABLE_WEIGHTS),
                          torch.from_numpy(LOCKED_FALLBACK_WEIGHTS))
    return SimpleNamespace(features=torch.stack(features, dim=1), logits=torch.stack(logits, dim=1),
                           targets=targets, ids=ids, reliable=reliable, weights=weights)


def fuse(logits, weights):
    return (logits[:, 0] * weights[:, :1] + logits[:, 1] * weights[:, 1:2]
            + logits[:, 2] * weights[:, 2:3])


def correction(module, features, reliable, head_weights):
    delta = module(features, reliable)
    residual = torch.einsum('bvd,vcd->bvc', delta, head_weights)
    mask = torch.stack([torch.zeros_like(reliable), torch.ones_like(reliable), reliable], dim=1)
    return residual * mask[..., None]


def fit(kind, batch, stage, heads, epochs):
    torch.manual_seed(1001 + stage)
    module = Exchange(batch.features.shape[-1], kind).cuda()
    optim = torch.optim.Adam(module.parameters(), lr=.0004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    rng = torch.Generator().manual_seed(1901 + stage)
    x, y, z, mask, w = [v.cuda() for v in
                         (batch.features[:, :, 0], batch.targets, batch.logits, batch.reliable, batch.weights)]
    for _ in range(epochs):
        for i in torch.randperm(len(x), generator=rng).split(64):
            i = i.cuda()
            output = fuse(z[i] + correction(module, x[i], mask[i], heads), w[i])
            loss = F.binary_cross_entropy_with_logits(output, y[i])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{kind} stage {stage}')
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()
    return module.eval()


@torch.no_grad()
def score(batch, modules, heads):
    pieces = []
    for start in range(0, len(batch.targets), 64):
        subset = slice(start, start + 64)
        x = batch.features[subset].cuda()
        reliable = batch.reliable[subset].cuda()
        delta = torch.cat([correction(module, x[:, :, t], reliable, heads[t])
                           for t, module in enumerate(modules)], dim=-1)
        pieces.append(fuse(batch.logits[subset].cuda() + delta, batch.weights[subset].cuda()).cpu())
    return torch.cat(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    logs = Path('logs/r1_final_cls') / root.name
    logs.mkdir(parents=True, exist_ok=True)
    inputs = SimpleNamespace(
        output=root, reference_root=Path('../emotic_benchmark_runs'),
        data_root=Path('../multi-lane-main/datasets/EMOTIC'),
        clip_checkpoint=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'),
        face_manifest_root=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'),
    )

    def train(view):
        with (logs / f'native_{view}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', '-m', 'multi_lane.track_a.runner',
                            *flags(inputs, 'native', view, args.smoke)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(3) as workers:
        list(workers.map(train, VIEWS))
    paths = [root / 'native' / view / 'train' for view in VIEWS]
    configs = [read(path / 'config.json') for path in paths]
    set_seed(1, True)
    models = [model_from_config(c, inputs.clip_checkpoint).eval() for c in configs]
    torch.set_num_threads(2)
    splits = ('train', 'val') if args.smoke else ('train', 'val', 'test')
    sources = {s: [dataset(c, s, inputs) for c in configs] for s in splits}
    comparisons = ('native', 'mlp', 'attention') if args.smoke else ('native', 'patch_pool', 'mlp', 'attention')
    rows = {s: {k: [[], [], []] for k in comparisons} for s in splits[1:]}
    modules = {k: [] for k in KINDS}
    heads = []
    for stage in range(2 if args.smoke else 8):
        for model, path in zip(models, paths):
            payload = torch.load(path / 'compact_checkpoints' / f'task{stage}.pth',
                                 map_location='cpu', weights_only=False)
            missing = model.load_state_dict(payload['model'], strict=False)
            if missing.unexpected_keys or any(not n.startswith('visual_encoder.') for n in missing.missing_keys):
                raise ValueError('Incomplete native state')
            model.restore_task(stage)
            model.requires_grad_(False)
        heads.append(torch.stack([m.head.weight[list(task_indices(stage))] for m in models]))
        training = extract(models, sources['train'], stage, 'train', args.smoke)
        for kind in KINDS:
            module = fit(kind, training, stage, heads[-1], 1 if args.smoke else 5)
            modules[kind].append(module)
            folder = root / kind
            folder.mkdir(exist_ok=True)
            torch.save(dict(model=module.state_dict(), stage=stage, cap=CAP), folder / f'task{stage}.pth')
        del training

        for split in splits[1:]:
            batch = extract(models, sources[split], stage, split, args.smoke)
            scores = {'native': fuse(batch.logits, batch.weights)}
            if not args.smoke:
                score_dir = 'val/scores' if split == 'val' else 'train/test_scores'
                def reference_logits(kind, view):
                    dump = load_evaluation_scores(args.reference / kind / view / score_dir / f'task{stage}.npz', 64)
                    index = {sample: i for i, sample in enumerate(dump.sample_ids)}
                    order = [index[sample] for sample in batch.ids]
                    if len(order) != len(index) or not np.array_equal(dump.targets[order], batch.targets.numpy()):
                        raise ValueError('Reference population mismatch')
                    return torch.from_numpy(dump.logits[order])
                for i, view in enumerate(VIEWS):
                    if not torch.equal(reference_logits('native', view), batch.logits[:, i]):
                        raise ValueError(f'Native {view} replay failed at {split} stage {stage}')
                logits = batch.logits.clone()
                logits[:, 2] = reference_logits('pool', 'face')
                scores['patch_pool'] = fuse(logits, batch.weights)
            scores.update({kind: score(batch, modules[kind], heads) for kind in KINDS})
            for kind, logits in scores.items():
                for group, mask in enumerate((torch.ones_like(batch.reliable), batch.reliable, ~batch.reliable)):
                    if mask.any():
                        rows[split][kind][group].append(compute_metrics(
                            stage, logits[mask].sigmoid(), batch.targets[mask], .5))
                print(split, stage, kind, rows[split][kind][0][-1].mAP, flush=True)
    results = {
        split: {kind: {group: dict(summary=summarize_tasks(values), stages=[v.mAP for v in values])
                       for group, values in zip(('all', 'reliable', 'unreliable'), groups)
                       if len(values) == (2 if args.smoke else 8)}
                for kind, groups in arms.items()}
        for split, arms in rows.items()
    }
    write(root / 'completion.json', dict(status='complete', smoke=args.smoke, results=results))


if __name__ == '__main__':
    main()
