"""TRAIN-only cohort allocation and protected sample-level fusion updates."""
from contextlib import nullcontext
from copy import copy
import hashlib

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import LabelView, compute_asymmetric_training_loss, task_indices


SAMPLE_MODES = ('sample_full', 'sample_holdout')
FUSION_ASL = dict(gamma_neg=9.8, gamma_pos=0., clip=.05, eps=1e-8)


def image_key(value):
    image, sep, person = str(value).rpartition('#person=')
    if not sep or not image.startswith('train:') or not person.isdecimal():
        raise ValueError('Expected an original TRAIN image and person index')
    return image


def global_image_roles(source, mode, seed):
    images = sorted({image_key(person) for person in source.sample_ids})
    if mode == 'sample_full':
        return {img: 'both' for img in images}
    if mode != 'sample_holdout':
        raise ValueError(f'Unexpected sample strategy {mode!r}')
    salt = f'd9-sample-fusion-{seed}:'
    ranked = sorted(images, key=lambda img: (hashlib.sha256((salt + img).encode()).digest(), img))
    count = (len(images) + 5) // 10
    if not 0 < count < len(images):
        raise ValueError('Original TRAIN is too small to partition 90/10')
    held = set(ranked[:count])
    return {img: 'fusion' if img in held else 'branch' for img in images}


def smoke_rows(source, indices):
    # Match the previous fixed smoke exactly in sample_full.
    by_mask = [[i for i in indices if bool(source.face_reliable[i]) == mask] for mask in (False, True)]
    return list(dict.fromkeys([*by_mask[0][:4], *by_mask[1][:4], *indices[:8]]))[:8]


def task_cohorts(source, indices, roles, smoke):
    branch, fusion = [], []
    for i in indices:
        role = roles[image_key(source.sample_ids[i])]
        if role in ('branch', 'both'):
            branch.append(i)
        if role in ('fusion', 'both'):
            fusion.append(i)
    if smoke:
        branch, fusion = smoke_rows(source, branch), smoke_rows(source, fusion)
    if not branch or not fusion:
        raise ValueError('Missing branch/fusion TRAIN cohort')
    return branch, fusion


def describe_cohort(source, indices, classes):
    ids = [str(source.sample_ids[i]) for i in indices]
    reliable = [i for i in indices if source.face_reliable[i]]
    return dict(people=len(ids), images=len({image_key(person) for person in ids}),
                sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                reliable_face=len(reliable), face_training_valid=sum(bool(source.face_training_valid[i]) for i in indices),
                classes=list(classes), positives=[sum(c in source.targets[i] for i in indices) for c in classes],
                reliable_face_positives=[sum(c in source.targets[i] for i in reliable) for c in classes])


def fusion_source(source):
    """Retain official TRAIN rows, boxes and labels; use deterministic evaluation crops."""
    gate_source = copy(source)
    gate_source.train = False
    gate_source.transforms_by_view = {v: (evaluation, evaluation)
                                      for v, (_, evaluation) in source.transforms_by_view.items()}
    return gate_source


def epoch_groups(size, updates, micro_batch_size, generator):
    if not 0 < updates <= size or micro_batch_size < 1:
        raise ValueError('Fusion sweep must include every person once and have nonempty optimizer steps')
    order = torch.randperm(size, generator=generator).tolist()
    groups = []
    for step in range(updates):
        rows = order[step * size // updates:(step + 1) * size // updates]
        groups.append([rows[k:k + micro_batch_size] for k in range(0, len(rows), micro_batch_size)])
    return groups


def train_gate_sweep(model, source, indices, task, epoch, config, opts):
    """At fixed branch weights, update each readout on the same people and exact step boundaries."""
    steps = config['gate_updates'] // config['epochs']
    generator = torch.Generator().manual_seed(800_000 + config['seed'] + 10_000 * task + epoch)
    groups = epoch_groups(len(indices), steps, config['micro_batch_size'], generator)
    microbatches = [rows for group in groups for rows in group]
    data = LabelView(source, indices, task_indices(task))
    loader = DataLoader(data, batch_sampler=microbatches, generator=generator,
                        num_workers=config['workers'], worker_init_fn=single_thread_worker, pin_memory=True,
                        **({'prefetch_factor': 1} if config['workers'] else {}))
    model.eval()
    precision = config['precision']
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    totals = {name: torch.zeros((), device='cuda') for name in opts}
    weights = {name: torch.zeros((2, 3), device='cuda') for name in opts}
    grad_max = {name: 0. for name in opts}
    sample_count = reliable_count = 0
    batches = iter(loader)
    for group in groups:
        logical_size = sum(map(len, group))
        for gate in opts.values():
            gate.reset_batch('cuda')
        for rows in group:
            inputs, labels = next(batches)
            if len(labels) != len(rows):
                raise RuntimeError('Incomplete gate microbatch')
            images = {key: item.cuda(non_blocking=True) for key, item in inputs.items()}
            target = labels.cuda(non_blocking=True).float()
            reliable = images['evidence_face_reliable']
            with torch.autocast('cuda', dtype=dtype) if precision != 'fp32' else nullcontext():
                outputs = model(images, sample_gate_only=True)
            for name, opt in opts.items():
                out = outputs['readouts'][name]
                logits = out['fused'].float()
                if name == 'mid_asl':
                    loss = compute_asymmetric_training_loss(logits, target, range(target.shape[1]),
                                                            1., 'current_only', **FUSION_ASL)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                weighted_loss = loss * (len(labels) / logical_size)
                torch.autograd.backward(weighted_loss, inputs=opt.params)
                opt.loss += weighted_loss.detach()
                opt.record_weights(out, reliable)
            sample_count += len(labels)
            reliable_count += int(reliable.sum())
        # No optimizer changes until all parallel readouts have passed the finite check.
        for opt in opts.values():
            opt.check_gradients(1.)
        for name, opt in opts.items():
            totals[name] += opt.loss * logical_size
            grad_max[name] = max(grad_max[name], opt.grad_norm.item())
            weights[name] += opt.weight_sums
            opt.step()
    if sample_count != len(indices):
        raise RuntimeError('Lost a fusion TRAIN person')
    reports = {}
    for name, opt in opts.items():
        means = [None if count == 0 else (val / count).tolist()
                 for val, count in zip(weights[name], (sample_count - reliable_count, reliable_count))]
        reports[name] = dict(objective='fusion_asl' if name == 'mid_asl' else 'fusion_bce',
                             mean_person_loss=(totals[name] / sample_count).item(),
                             gradient_max=grad_max[name], gate_updates=steps, gate_total_updates=opt.updates,
                             mean_preupdate_weights_by_reliability=means)
    return reports, dict(people=sample_count, reliable_face=reliable_count, updates_this_epoch=steps)
