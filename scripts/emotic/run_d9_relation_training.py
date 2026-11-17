"""Joint training of three D9 lanes and intermediate-relation logit weights."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.d9_relation_fusion import D9RelationFusion
from multi_lane.track_a.r1_crossvit import VIEWS, compact, restore_rng, rng_state
from multi_lane.track_a.r1_crossvit_experiment import (
    model_from_config, read, single_thread_worker, source_runs, write)
from multi_lane.track_a.runner import (
    LabelView, TASK_SIZES, build_optimizer_groups, compute_asymmetric_training_loss,
    compute_training_loss, fit_calibration_indices, git_metadata, set_seed, task_indices)
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def branch_configs(inputs):
    result = {}
    for view, path in source_runs(inputs.reference_root).items():
        source = read(path / 'config.json')
        if source['seed'] != 1 or source['adapter_residual_scale'] != .1 or source['parameter_group_loss_routing'] != 'adapter_asl':
            raise ValueError(f'Unexpected source recipe in {path}')
        result[view] = {key: source[key] for key in ('num_selectors', 'num_prompts', 'num_prompt_layers', 'normalize')}
        result[view].update(r1_receiver=view, r1_kind='native' if view == 'full' else 'attention',
                            r1_frozen_mode=None if view == 'full' else 'patch', r1_exchange_layer=8,
                            r1_frozen_cap=.05, r1_frozen_evidence='all')
    return result


def make_model(config, device='cuda'):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('The historical CLIP model constructor requires a CUDA device')
    branches = {}
    with torch.cuda.device(device if device.index is not None else torch.cuda.current_device()):
        for view in VIEWS:
            # Each historical branch was independently initialized under the same seed.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(config['seed'])
                branch = model_from_config(config['branches'][view], Path(config['clip_checkpoint']))
            if branches:
                branch.visual_encoder = branches['full'].visual_encoder
            branches[view] = branch
        return D9RelationFusion(branches, config['mode']).to(device)


def snapshot(model):
    return {'branches': {view: compact(branch) for view, branch in model.branches.items()},
            'gates': {key: val.detach().cpu() for key, val in model.gates.state_dict().items()}}


def restore(model, state, task):
    for view, branch in model.branches.items():
        mismatch = branch.load_state_dict(state['branches'][view], strict=False)
        if mismatch.unexpected_keys or any(not key.startswith('visual_encoder.') for key in mismatch.missing_keys):
            raise ValueError(f'{view} method parameters differ from checkpoint')
    model.gates.load_state_dict(state['gates'], strict=True)
    model.restore_task(task)


def save_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_state(path, config, task=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config'] != config or (task is not None and task != state['task']):
        raise ValueError(f'Incompatible checkpoint {path}')
    return state


def load_evaluation_model(root, task, device='cuda'):
    root = Path(root)
    config = read(root / 'config.json')
    model = make_model(config, device)
    state = read_state(root / 'compact_checkpoints' / f'task{task}.pth', config, task)
    restore(model, state, task)
    return model.eval()


def optimizer_for_task(model, config, task):
    primary, adapters, groups = [], [], []
    for branch in model.branches.values():
        base, adapted, branch_groups = build_optimizer_groups(branch, 0., adapter_learning_rate=.0004)
        primary.extend(base)
        adapters.extend(adapted)
        groups.extend(branch_groups)
    # In multi mode the branch optimizer must exactly match protected fixed.
    gate = [] if config['mode'] == 'multi' else list(model.gates[task].parameters())
    if gate:
        groups.append({'params': gate, 'lr': config['gate_lr'], 'weight_decay': 0.})
        primary.extend(gate)
    routed = [id(p) for p in (*primary, *adapters)]
    optimized = [id(p) for g in groups for p in g['params']]
    if len(set(routed)) != len(routed) or sorted(routed) != sorted(optimized):
        raise RuntimeError('Optimizer and loss routing parameter sets differ')
    optimizer = torch.optim.Adam(groups, lr=.0125, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    return primary, adapters, optimizer, scheduler


class GateOptimizer:
    """Private Adam and training diagnostics for one current-task readout."""

    def __init__(self, gate, config):
        self.params = list(gate.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=config['gate_lr'], weight_decay=0.)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])

    def reset_batch(self, device):
        self.optimizer.zero_grad(set_to_none=True)
        self.loss = torch.zeros((), device=device)
        self.weight_sums = torch.zeros((2, 3), device=device)

    def accumulate(self, output, target, reliable, total_rows, scale):
        loss = F.binary_cross_entropy_with_logits(output['fused'].float(), target)
        loss = loss * (len(target) / total_rows)
        torch.autograd.backward(loss * scale, inputs=self.params)
        self.loss += loss.detach()
        weights = output['weights'][:, 0].detach()
        self.weight_sums[0] += weights[~reliable].sum(0)
        self.weight_sums[1] += weights[reliable].sum(0)

    def check_gradients(self, scale):
        if any(p.grad is None for p in self.params):
            raise RuntimeError('Trainable gate parameter was disconnected')
        grads = [p.grad for p in self.params]
        if scale != 1.:
            torch._foreach_mul_(grads, 1. / scale)
        self.grad_norm = torch.stack(torch._foreach_norm(grads)).norm()
        if not torch.isfinite(self.grad_norm) or not torch.isfinite(self.loss):
            raise FloatingPointError('Nonfinite gate loss or gradient')

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])


def losses(outputs, labels, valid, total_rows, valid_rows, classes, *, protect_branches=False):
    # Normalize each loss by its cohort over the entire logical batch. Gradients
    # of the accumulated microbatches then equal gradients of the unsplit batch.
    fused_logits = outputs['fused'].detach() if protect_branches else outputs['fused']
    fused = F.binary_cross_entropy_with_logits(fused_logits.float(), labels) * (len(labels) / total_rows)
    branch_bce, adapter_asl = [], []
    for view in VIEWS:
        logits, target = outputs['all_columns'][view], labels
        if view == 'face':
            logits, target = logits[valid], target[valid]
            if target.shape[0] == 0:
                continue
        factor = len(target) / (valid_rows if view == 'face' else total_rows)
        branch_bce.append(factor * compute_training_loss(logits, target, classes, 1., 'legacy_full_zero'))
        adapter_asl.append(factor * compute_asymmetric_training_loss(logits, target, classes, 1., 'legacy_full_zero'))
    return fused, sum(branch_bce), sum(adapter_asl)


def update_batch(model, batch, classes, config, primary, adapters, optimizer, precision, gate_opts):
    images, labels = batch
    count = len(labels)
    for gate in gate_opts.values():
        gate.reset_batch('cuda')
    valid_count = int(images['face_training_valid'].sum())
    reliable_count = int(images['evidence_face_reliable'].sum())
    optimizer.zero_grad(set_to_none=True)
    # Hard failure on nonfinite gradients precedes any update. A failed half-
    # precision batch can be replayed with the same tensors and FP32 arithmetic.
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    scale = 4096. if precision == 'fp16' else 1.
    totals = torch.zeros(3, device='cuda')
    weight_sum = torch.zeros((2, 3), device='cuda')
    for start in range(0, count, config['micro_batch_size']):
        stop = start + config['micro_batch_size']
        piece = {k: value[start:stop].cuda(non_blocking=True) for k, value in images.items()}
        target = labels[start:stop].cuda(non_blocking=True).float()
        with torch.autocast('cuda', dtype=dtype) if precision != 'fp32' else nullcontext():
            outputs = model(piece)
            fused, bce, asl = losses(
                outputs, target, piece['face_training_valid'], count, valid_count, classes,
                protect_branches=config.get('protect_branches', False),
            )
        # backward(inputs=...) accumulates into .grad across all microbatches.
        torch.autograd.backward((bce + fused) * scale, inputs=primary, retain_graph=True)
        torch.autograd.backward((asl + fused) * scale, inputs=adapters)
        totals += torch.stack((fused.detach(), bce.detach(), asl.detach()))
        reliable = piece['evidence_face_reliable']
        for name, gate in gate_opts.items():
            gate.accumulate(outputs['readouts'][name], target, reliable, count, scale)
        weights = outputs['weights'][:, 0].detach()
        weight_sum[0] += weights[~reliable].sum(0)
        weight_sum[1] += weights[reliable].sum(0)
    grads = [p.grad for p in (*primary, *adapters) if p.grad is not None]
    if not grads:
        raise RuntimeError('No joint gradients')
    if scale != 1.:
        torch._foreach_mul_(grads, 1. / scale)
    norms = torch.stack(torch._foreach_norm(grads))
    if not torch.isfinite(norms).all() or not torch.isfinite(totals).all():
        raise FloatingPointError('Nonfinite pre-update values')
    # Check every gradient before applying any of the optimizers. A retry
    # starts with the original branch and gate states and the same input batch.
    for gate in gate_opts.values():
        gate.check_gradients(scale)
    gate_grad = [] if gate_opts else [
        p.grad for p in model.gates[model.current_task_id].parameters() if p.grad is not None]
    gate_norm = torch.stack(torch._foreach_norm(gate_grad)).norm().item() if gate_grad else 0.
    optimizer.step()
    for gate in gate_opts.values():
        gate.optimizer.step()
    return totals.tolist(), weight_sum.tolist(), count, valid_count, reliable_count, gate_norm


def train_epoch(model, source, indices, task, epoch, config, primary, adapters, optimizer, gate_opts):
    classes = task_indices(task)
    seed = config['seed'] + task * 10_000 + epoch
    generator = torch.Generator().manual_seed(seed)
    set_seed(seed, tf32=True)
    data = LabelView(source, indices, classes)
    kwargs = {'prefetch_factor': 1} if config['workers'] else {}
    loader = DataLoader(data, batch_size=config['batch_size'], shuffle=True, generator=generator,
                        num_workers=config['workers'], pin_memory=True, worker_init_fn=single_thread_worker, **kwargs)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    train_losses = [0., 0., 0.]
    gate_sums = [[0., 0., 0.], [0., 0., 0.]]
    seen = valid = reliable = retries = updates = 0
    gate_grad_max = 0.
    learned_losses = {name: 0. for name in gate_opts}
    learned_grad_max = {name: 0. for name in gate_opts}
    learned_sums = {name: torch.zeros((2, 3), dtype=torch.float64, device='cuda') for name in gate_opts}
    for batch in loader:
        try:
            result = update_batch(model, batch, classes, config, primary, adapters, optimizer, config['precision'], gate_opts)
        except FloatingPointError:
            if config['precision'] == 'fp32':
                raise
            optimizer.zero_grad(set_to_none=True)
            retries += 1
            print(f'task {task} epoch {epoch} update {updates}: replaying this batch in FP32', flush=True)
            result = update_batch(model, batch, classes, config, primary, adapters, optimizer, 'fp32', gate_opts)
        for name, gate in gate_opts.items():
            learned_losses[name] += gate.loss.item()
            learned_sums[name] += gate.weight_sums
            learned_grad_max[name] = max(learned_grad_max[name], gate.grad_norm.item())
        batch_losses, weight_sum, n, v, r, gate_grad = result
        gate_grad_max = max(gate_grad_max, gate_grad)
        seen += n
        valid += v
        reliable += r
        updates += 1
        for i, loss in enumerate(batch_losses):
            train_losses[i] += loss
        for i, row in enumerate(weight_sum):
            gate_sums[i] = [a+b for a, b in zip(gate_sums[i], row)]
    if seen != len(indices):
        raise RuntimeError('Lost a training row')
    report = {'task': task, 'epoch': epoch, 'examples': seen, 'valid_face': valid, 'reliable_face': reliable,
            'updates': updates, 'skipped': 0, 'fp32_retries': retries,
            'gate_gradient_max': gate_grad_max, 'mean_view_weights_by_reliability': [
                [s/n for s in row] if n else None for row, n in zip(gate_sums, (seen-reliable, reliable))],
            'mean_batch_losses': dict(zip(('fused_bce', 'branch_bce_sum', 'adapter_asl_sum'), [s/updates for s in train_losses])),
            'seconds': time.monotonic()-start_time, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
    if gate_opts:
        report['train_readouts'] = {
            name: {'mean_batch_bce': learned_losses[name] / updates,
                   'gradient_max': learned_grad_max[name],
                   'mean_preupdate_weights_by_reliability': [
                       [s/n for s in row] if n else None
                       for row, n in zip(learned_sums[name].tolist(), (seen-reliable, reliable))]}
            for name in gate_opts}
    return report


def train(args):
    root = args.output
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    inputs = input_paths()
    precision = args.precision
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp32'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support BF16')
    config = dict(version=2, deterministic_algorithms=True, mode=args.mode, seed=1, epochs=1 if args.smoke else 30,
                  tasks=2 if args.smoke else len(TASK_SIZES), batch_size=4 if args.smoke else 64,
                  micro_batch_size=args.micro_batch_size, workers=args.workers, gate_lr=.0004,
                  precision=precision, smoke=args.smoke, clip_checkpoint=str(inputs.clip_checkpoint),
                  branches=branch_configs(inputs))
    if args.protect_branches or args.mode == 'multi':
        config['protect_branches'] = True
    if (root/'config.json').exists() and read(root/'config.json') != config:
        raise ValueError('Existing experiment has a different configuration')
    write(root/'config.json', config)
    source = build_joint_source(inputs, 'train')
    set_seed(config['seed'], tf32=True)
    model = make_model(config)
    last = root/'epoch_recovery'/'latest.pth'
    checkpoint_dir = root/'compact_checkpoints'
    completed = [p for p in range(config['tasks']) if (checkpoint_dir/f'task{p}.pth').is_file()]
    if completed != list(range(len(completed))):
        raise RuntimeError('Completed task checkpoints are not contiguous')
    if completed:
        state = read_state(checkpoint_dir/f'task{completed[-1]}.pth', config, completed[-1])
        restore(model, state, completed[-1])
        restore_rng(state['rng'])
        del state
    print(f"mode={args.mode} precision={precision} completed_tasks={len(completed)}", flush=True)
    for task in range(len(completed), config['tasks']):
        model.activate_task(task)
        primary, adapters, optimizer, scheduler = optimizer_for_task(model, config, task)
        gate_opts = {}
        if config['mode'] == 'multi':
            gate_opts = {name: GateOptimizer(model.gates[task][name], config)
                         for name in ('static', 'relation', 'summary')}
        indices, _ = fit_calibration_indices(source, task_indices(task), 0.)
        if args.smoke:
            # Include both confidence cohorts when available.
            by_mask = [[i for i in indices if bool(source.face_reliable[i]) == m] for m in (False, True)]
            indices = list(dict.fromkeys([*by_mask[0][:4], *by_mask[1][:4], *indices[:8]]))[:8]
        coverage = dict(count=len(indices), face_training_valid=sum(bool(source.face_training_valid[i]) for i in indices),
                        reliable_face=sum(bool(source.face_reliable[i]) for i in indices),
                        sample_id_sha256=hashlib.sha256('\n'.join(str(source.sample_ids[i]) for i in indices).encode()).hexdigest())
        history = []
        if last.is_file():
            state = read_state(last, config)
            if state['task'] == task:
                if state['coverage'] != coverage:
                    raise ValueError('TRAIN cohort changed since recovery')
                restore(model, state, task)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                for name, gate in gate_opts.items():
                    gate.load_state_dict(state['gate_optimizers'][name])
                restore_rng(state['rng'])
                history = state['history']
            elif state['task'] != task - 1:
                raise RuntimeError('Epoch recovery does not match completed task chain')
            del state
        for epoch in range(len(history)+1, config['epochs']+1):
            row = train_epoch(model, source, indices, task, epoch, config, primary, adapters, optimizer, gate_opts)
            history.append(row)
            scheduler.step()
            for gate in gate_opts.values():
                gate.scheduler.step()
            state = dict(config=config, task=task, coverage=coverage, history=history, rng=rng_state(), **snapshot(model))
            recovery = {**state, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            if gate_opts:
                recovery['gate_optimizers'] = {name: gate.state_dict() for name, gate in gate_opts.items()}
            save_atomic(last, recovery)
            del recovery
            write(root/'progress.json', row)
            print(json.dumps(row), flush=True)
        # Also handles recovery after the last epoch was saved but before task publication.
        state = dict(config=config, task=task, coverage=coverage, history=history, rng=rng_state(), **snapshot(model))
        save_atomic(checkpoint_dir/f'task{task}.pth', state)
        write(root/'history'/f'task{task}.json', {'coverage': coverage, 'epochs': history})
        del state, optimizer, scheduler
    write(root/'training_complete.json', dict(status='complete', tasks=config['tasks'], smoke=args.smoke,
                                              git=git_metadata(Path.cwd())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('fixed', 'static', 'person', 'multi'))
    parser.add_argument('--protect-branches', action='store_true',
                        help='Block fusion gradients into the three branches (automatic in multi mode).')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--micro-batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32', 'fp16'), default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.protect_branches and args.mode not in ('fixed', 'multi'):
        parser.error('--protect-branches applies to fixed or multi mode')
    if args.micro_batch_size < 1 or args.workers < 0:
        parser.error('micro-batch-size must be positive; workers must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    with (args.output/'.train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args)


if __name__ == '__main__':
    main()
