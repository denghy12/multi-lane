"""Extra held-TRAIN readouts at the original branch-update clock.

Only the evidence source differs: 90% branches never update on these held rows.
Global/summary get one update per branch batch; bounded/direction get 300/task.
"""
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_direction_objective import deranged_targets, direction_benefits, gate_update_due
from multi_lane.track_a.r1_crossvit import VIEWS, rng_state, restore_rng
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import LabelView, task_indices


EVERY_BATCH = ('global_relation', 'task_summary', 'shared_static')
DIRECTION_NAMES = ('own_benefit', 'shuffled_benefit')


def held_batches(size, count, batch_size, seed):
    """Independent uniform held batches, without duplicate identities within a step."""
    if min(size, count, batch_size) < 1:
        raise ValueError('Expected nonempty held data and training budget')
    generator = torch.Generator().manual_seed(seed)
    return [torch.randperm(size, generator=generator)[:min(batch_size, size)].tolist()
            for _ in range(count)]


class HeldGateUpdates:
    def __init__(self, model, source, indices, task, epoch, batches, config, opts):
        self.model, self.config, self.opts = model, config, opts
        self.task, self.epoch, self.batches = task, epoch, batches
        seed = 31_000_000 + config['seed'] + task * 100_000 + epoch
        sampler = held_batches(len(indices), batches, config['batch_size'], seed)
        generator = torch.Generator().manual_seed(seed + 1)
        self.loader = DataLoader(LabelView(source, indices, task_indices(task)), batch_sampler=sampler,
                                 generator=generator, num_workers=config['workers'], pin_memory=True,
                                 worker_init_fn=single_thread_worker,
                                 **({'prefetch_factor': 1} if config['workers'] else {}))
        self.iterator = iter(self.loader)
        self.stats = {name: dict(updates=0, loss=0., people=0, gradient_max=0., positives=[0, 0], valid=[0, 0])
                      for name in opts}

    def step(self, batch_index):
        due = gate_update_due(batch_index, self.batches, self.epoch, self.config['epochs'],
                              self.config['gate_updates'])
        selected = {n: o for n, o in self.opts.items() if n in EVERY_BATCH or due}
        batch = next(self.iterator)
        if not selected:
            return
        state, training = rng_state(), self.model.training
        try:
            self.model.eval()
            inputs, labels = batch
            images = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
            target = labels.cuda(non_blocking=True).float()
            reliable = images['evidence_face_reliable'].bool()
            for opt in selected.values():
                opt.reset_batch('cuda')
            precision = self.config['precision']
            dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
            with torch.autocast('cuda', dtype=dtype) if precision != 'fp32' else nullcontext():
                out = self.model(images, sample_gate_only=True)
            targets = shuffled = usable = None
            if any(n in selected for n in DIRECTION_NAMES):
                branches = torch.stack([out['branches'][v].detach() for v in VIEWS], -1)
                benefit, valid = direction_benefits(branches, target, reliable, .05)
                targets = (benefit > 0).float()
                shuffled, usable, _ = deranged_targets(targets, valid, reliable,
                    self.config['seed'] + self.task * 10_000_000 + self.epoch * 100_000 + batch_index)
                if not torch.equal((targets * usable).sum(0), (shuffled * usable).sum(0)):
                    raise RuntimeError('Shuffling changed the direction-label counts')
            for name, opt in selected.items():
                readout = out['readouts'][name]
                if name in DIRECTION_NAMES:
                    opt.accumulate_direction(readout, reliable)
                    opt.backward_direction(targets if name == 'own_benefit' else shuffled, usable, 1.)
                else:
                    opt.accumulate(readout, target, reliable, len(target), 1.)
                opt.check_gradients(1.)
            # All objectives must be finite before changing any gate.
            for name, opt in selected.items():
                opt.step()
                stats = self.stats[name]
                stats['updates'] += 1
                stats['loss'] += opt.loss.item()
                stats['people'] += len(target)
                stats['gradient_max'] = max(stats['gradient_max'], opt.grad_norm.item())
                stats['positives'] = [a + int(b) for a, b in zip(stats['positives'], opt.positive_counts.tolist())]
                stats['valid'] = [a + int(b) for a, b in zip(stats['valid'], opt.supervision_counts.tolist())]
        finally:
            self.model.train(training)
            restore_rng(state)

    def report(self):
        return {name: dict(**stats, mean_loss=stats['loss'] / max(stats['updates'], 1),
                           total_updates=self.opts[name].updates)
                for name, stats in self.stats.items()}
