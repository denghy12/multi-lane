"""Read-only causal rollback of shared experts across task boundaries.

All expert patches live in memory and are restored in finally blocks. Support
is selected on fixed TRAIN batches only. Both endpoints execute the same old
lanes; current-task performance is deliberately outside this attribution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .gradient_attribution import grouped_batches
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import (TASK_SIZES, LabelView, compact_model_state_dict, compute_metrics,
                     _intersects, git_metadata, seen_indices, set_seed, tensor_state_digest)

CONDITIONS = ('before', 'after', 'rollback_accessed', 'rollback_frozen', 'rollback_all_mutable')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def expert_keys(state, pairs):
    prefixes = tuple(f'adapter_bank.experts.{l}.{e}.' for l, e in pairs)
    return [k for k in state if k.startswith(prefixes)] if prefixes else []


@contextmanager
def rollback_experts(model, reference, pairs):
    """Patch whole experts (including biases), leaving all other state intact."""
    state = model.state_dict()
    keys = expert_keys(state, pairs)
    if len(keys) != 4 * len(set(pairs)):
        raise ValueError('Each requested expert must have down/up weight and bias')
    backup = {k: state[k].detach().clone() for k in keys}
    unchanged = {k: v for k, v in state.items() if k not in keys}
    protected = tensor_state_digest(unchanged)
    try:
        with torch.no_grad():
            for k in keys:
                if reference[k].shape != state[k].shape:
                    raise ValueError('Expert shape mismatch: ' + k)
                state[k].copy_(reference[k])
        yield keys
        assert tensor_state_digest(unchanged) == protected, 'Nonexpert state mutated'
    finally:
        with torch.no_grad():
            for k in keys:
                state[k].copy_(backup[k])


def old_coordinate_digest(state, boundary):
    """Only coordinates used by the old task functions; excludes shared pool."""
    count = sum(TASK_SIZES[:boundary])
    old = {}
    for key, value in state.items():
        if key == 'selectors': old[key] = value[:boundary]
        elif key.startswith('prompts.'): old[key] = value[:, :boundary]
        elif key.startswith('head.'): old[key] = value[:count]
        elif key.startswith('adapter_bank.tasks.'):
            if int(key.split('.')[2]) < boundary: old[key] = value
        elif key == 'task_class_mask': old[key] = value
    if not old or not any(k.startswith('head.') for k in old):
        raise ValueError('Missing old function coordinates')
    return tensor_state_digest(old)


class SupportCapture:
    def __init__(self, bank, reliable):
        self.bank, self.reliable, self.pairs = bank, reliable, set()
        self.original = bank._route
        self.lookup = {id(pool): (t, l) for t, pools in enumerate(bank.tasks)
                       for l, pool in enumerate(pools)}

    def __call__(self, cls, pool, view):
        gate = self.original(cls, pool, view)
        _, layer = self.lookup[id(pool)]
        valid = self.reliable if view == 2 else torch.ones(len(cls), device=cls.device, dtype=torch.bool)
        selected = torch.nonzero((gate[valid] > 0).any(0)).flatten().tolist()
        self.pairs.update((layer, e) for e in selected)
        return gate


@torch.no_grad()
def forward(model, images, amp=False, collect=False):
    bank = model.adapter_bank
    capture = SupportCapture(bank, images['face_reliable'].bool()) if collect else None
    if capture: bank._route = capture
    try:
        with torch.autocast(device_type=next(model.parameters()).device.type, enabled=amp):
            fused, views = model.encode_lanes_with_views(images, all_seen_lanes=True)
            lane_ids = model._lane_ids(all_seen_lanes=True)
            logits, view_logits = model._lane_logits_with_views(fused, views, images, lane_ids)
            mask = model.task_class_mask[lane_ids].to(logits.dtype)
            combine = lambda x: (x * mask.unsqueeze(0)).sum(1)[:, :model.seen_classes].float().cpu()
            result = {'feature_fused': fused.float().cpu(), 'logit_fused': combine(logits)}
            for name in views:
                result['feature_' + name] = views[name].float().cpu()
                result['logit_' + name] = combine(view_logits[name])
    finally:
        if capture: bank._route = capture.original
    return result, capture.pairs if capture else set()


def differences(left, right, reliable=None):
    result = {}
    for key in left:
        a, b = left[key], right[key]
        if key.endswith('_face') and reliable is not None:
            a, b = a[reliable], b[reliable]
        delta = a.double() - b.double()
        result[key] = dict(mean_abs=float(delta.abs().mean()) if delta.numel() else None,
                          max_abs=float(delta.abs().max()) if delta.numel() else None,
                          relative_l2=float(delta.norm()/b.double().norm().clamp_min(1e-12)) if delta.numel() else None)
    return result


def build(config, visual):
    names = ('num_selectors', 'num_prompts', 'num_prompt_layers', 'selector_mode',
             'prompt_mode', 'normalize', 'adapter_mode', 'adapter_bottleneck_dim',
             'adapter_layer_indices', 'adapter_residual_scale', 'moe_num_experts',
             'moe_top_k', 'moe_dropout', 'moe_noisy_gating', 'view_fusion', 'view_classifier_mode')
    return MultiLaneModel(visual, TASK_SIZES,
                         moe_freeze_metric=config.get('moe_freeze_metric') or 'hard_count',
                         **{k: config[k] for k in names}).float()


def load(model, payload, config, boundary):
    restore_compact_model_state(model, payload, payload['task_id'], config['git'])
    # Running exactly the same number of old lanes removes the changing-GEMM
    # shape confound. Pool masks only govern training, never inference here.
    model.restore_task(boundary - 1)
    model.requires_grad_(False)
    model.eval()


def select_batches(source, boundary, count):
    classes = seen_indices(boundary - 1)
    eligible = [i for i, y in enumerate(source.targets) if _intersects(y, classes)]
    groups = grouped_batches([source.sample_ids[i] for i in eligible], 64, count, 20261018 + boundary)
    return [[eligible[j] for j in group] for group in groups]


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    set_seed(0, tf32=False)
    args.output.mkdir(parents=True)
    provenance = dict(status='running', git=git_metadata(Path(__file__).resolve().parents[2]), optimizer_steps=0,
                      source_checkpoints={}, conditions=list(CONDITIONS), boundaries=[1, 7],
                      validation_old_classes_only=True, test_loaded=False,
                      support_split='train', support_batch_cap=64, support_batches=args.train_batches,
                      source_mutations=0, settings=vars(args).copy())
    provenance['settings'] = {k: str(v) if isinstance(v, Path) else v for k, v in provenance['settings'].items()}
    write(args.output/'run.json', provenance)
    visual = load_openai_clip_visual(args.clip_checkpoint)
    visual_hash = tensor_state_digest(visual.state_dict())
    for method, root in (('AF', args.af), ('WAF', args.waf)):
        config = read(root/'config.json')
        assert config['adapter_mode'] == 'moe_task_ffn_activate_freeze'
        assert config['adapter_layer_indices'] == [1] and config['view_classifier_mode'] == 'shared_post_fusion'
        assert config['max_tasks'] == 8 and config['seed'] == 0 and not config['git']['dirty']
        source_sha = read(root.parent/'full_val/result.json')['checkpoint_sha256']
        train_source = make_source(config, args.data_root, args.face_manifest_root, 'train')
        val_source = make_source(config, args.data_root, args.face_manifest_root, 'val')
        model = build(config, visual).cuda()
        for boundary in (1, 7):
            start = time.time()
            directory = args.output/method/f'task{boundary}'
            directory.mkdir(parents=True)
            files = [root/'compact_checkpoints'/f'task{t}.pth' for t in (boundary-1, boundary)]
            hashes = [_sha256(p) for p in files]
            assert all(h == source_sha[str(t)] for h, t in zip(hashes, (boundary-1, boundary)))
            before, after = [_torch_load(p) for p in files]
            old_hash = old_coordinate_digest(before['model'], boundary)
            assert old_hash == old_coordinate_digest(after['model'], boundary), 'Old nonexpert coordinates changed'
            frozen = before['model']['adapter_bank.frozen_experts'].bool()
            all_frozen = [(l, e) for l in range(len(frozen)) for e in range(frozen.shape[1]) if frozen[l, e]]
            all_mutable = [(l, e) for l in range(len(frozen)) for e in range(frozen.shape[1]) if not frozen[l, e]]
            for key in expert_keys(before['model'], all_frozen):
                assert torch.equal(before['model'][key], after['model'][key]), key
            indices = select_batches(train_source, boundary, args.train_batches)
            batches = []
            for ids in indices:
                loader = DataLoader(LabelView(train_source, ids, seen_indices(boundary-1), include_sample_id=True),
                                    batch_size=len(ids), shuffle=False, num_workers=0)
                pixels, target, stable = next(iter(loader))
                batches.append((pixels, target, list(stable)))
            support = set()
            for payload in (before, after):
                load(model, payload, config, boundary)
                for amp in (False, True):
                    torch.backends.cuda.matmul.allow_tf32 = amp
                    torch.backends.cudnn.allow_tf32 = amp
                    for pixels, _, _ in batches:
                        _, accessed = forward(model, move_model_inputs(pixels, 'cuda'), amp=amp, collect=True)
                        support.update(accessed)
            selected = sorted(support.intersection(all_mutable))
            patches = {'before': [], 'after': [], 'rollback_accessed': selected,
                       'rollback_frozen': all_frozen, 'rollback_all_mutable': all_mutable}
            details = dict(boundary=boundary, method=method, old_classes=sum(TASK_SIZES[:boundary]),
                           checkpoint_sha256=hashes, old_nonexpert_sha256=old_hash,
                           support=sorted(support), patch_pairs=patches,
                           train_batches=[dict(sample_ids=b[2], pixel_sha256=tensor_state_digest(b[0]),
                                               samples=len(b[2])) for b in batches], train={})
            # FP32 primary and archived AMP/TF32 sensitivity, identical pixels.
            for amp in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = amp
                torch.backends.cudnn.allow_tf32 = amp
                precision = 'amp_tf32' if amp else 'fp32'
                baseline = {}
                for condition in CONDITIONS:
                    load(model, before if condition == 'before' else after, config, boundary)
                    initial = tensor_state_digest(compact_model_state_dict(model))
                    outputs = []
                    with rollback_experts(model, before['model'], patches[condition]):
                        patched_hash = tensor_state_digest(compact_model_state_dict(model))
                        for pixels, _, _ in batches:
                            value, _ = forward(model, move_model_inputs(pixels, 'cuda'), amp=amp)
                            outputs.append(value)
                        assert patched_hash == tensor_state_digest(compact_model_state_dict(model))
                    assert initial == tensor_state_digest(compact_model_state_dict(model))
                    baseline[condition] = outputs
                rows = []
                for i, (pixels, _, _) in enumerate(batches):
                    row = {c: differences(baseline[c][i], baseline['before'][i], pixels['face_reliable'].bool()) for c in CONDITIONS}
                    assert all(torch.equal(baseline['after'][i][k], baseline['rollback_frozen'][i][k]) for k in baseline['after'][i])
                    rows.append(row)
                details['train'][precision] = rows
                arrays = {f'{c}_batch{i}_{key}': value.numpy() for c in CONDITIONS for i, values in enumerate(baseline[c]) for key, value in values.items()}
                np.savez_compressed(directory/f'train_{precision}.npz', **arrays)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            validation = {}
            val_ids = list(range(len(val_source)))[:args.val_limit] if args.val_limit else list(range(len(val_source)))
            loader = DataLoader(LabelView(val_source, val_ids, seen_indices(boundary-1), include_sample_id=True),
                                batch_size=64, shuffle=False, num_workers=2)
            for condition in CONDITIONS:
                load(model, before if condition == 'before' else after, config, boundary)
                initial = tensor_state_digest(compact_model_state_dict(model))
                logits = {v: [] for v in ('fused', 'full', 'person', 'face')}
                targets, stable_ids, reliability = [], [], []
                with rollback_experts(model, before['model'], patches[condition]):
                    patched_hash = tensor_state_digest(compact_model_state_dict(model))
                    for pixels, target, stable in loader:
                        value, _ = forward(model, move_model_inputs(pixels, 'cuda'))
                        for view in logits: logits[view].append(value['logit_' + view])
                        targets.append(target); stable_ids.extend(stable)
                        reliability.append(pixels['face_reliable'].bool())
                    assert patched_hash == tensor_state_digest(compact_model_state_dict(model))
                assert initial == tensor_state_digest(compact_model_state_dict(model))
                branch = {v: torch.cat(parts) for v, parts in logits.items()}
                z, y = branch['fused'], torch.cat(targets)
                reliable = torch.cat(reliability)
                validation[condition] = asdict(compute_metrics(boundary-1, z.sigmoid(), y, .5))
                validation[condition]['view_metrics'] = {v: asdict(compute_metrics(boundary-1, values.sigmoid(), y, .5))
                                                         for v, values in branch.items()}
                if bool(reliable.any()):
                    validation[condition]['view_metrics']['face_reliable'] = asdict(compute_metrics(
                        boundary-1, branch['face'][reliable].sigmoid(), y[reliable], .5))
                np.savez_compressed(directory/f'val_{condition}.npz', logits=z.numpy(), probabilities=z.sigmoid().numpy(),
                                    targets=y.numpy(), sample_ids=np.asarray(stable_ids), class_indices=np.asarray(seen_indices(boundary-1)),
                                    face_reliable=reliable.numpy(), **{v+'_logits': a.numpy() for v, a in branch.items()})
                print(f'{method} boundary={boundary} {condition} old_val_mAP={validation[condition]["mAP"]:.6f}', flush=True)
            a = np.load(directory/'val_after.npz'); b = np.load(directory/'val_rollback_frozen.npz')
            assert np.array_equal(a['logits'], b['logits']), 'Negative control changed validation'
            details.update(validation=validation, elapsed_seconds=time.time()-start,
                           restored_model_unchanged=True, source_checkpoint_sha256_after=[_sha256(p) for p in files])
            assert hashes == details['source_checkpoint_sha256_after']
            assert all(p.grad is None for p in model.parameters())
            write(directory/'result.json', details)
            provenance['source_checkpoints'][f'{method}_{boundary}'] = hashes
            print(f'BOUNDARY_COMPLETE {method} {boundary} mutable_accessed={selected}', flush=True)
        assert tensor_state_digest(model.visual_encoder.state_dict()) == visual_hash
        del model
        torch.cuda.empty_cache()
    provenance.update(status='complete', frozen_visual_unchanged=True)
    write(args.output/'run.json', provenance)
    print('EXPERT_ROLLBACK_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--af', type=Path, required=True)
    parser.add_argument('--waf', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--face-manifest-root', type=Path, required=True)
    parser.add_argument('--clip-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-batches', type=int, choices=(1, 4), default=4)
    parser.add_argument('--val-limit', type=int, choices=(0, 64), default=0)
    args = parser.parse_args()
    if (args.train_batches == 1) != (args.val_limit == 64):
        parser.error('Smoke uses one TRAIN batch and 64 validation people together')
    run(args)


if __name__ == '__main__':
    main()
