"""Learn task-private logit weights from patch correspondences at the frozen receiver site."""

import math

import torch
from torch import nn

from .r1_crossvit import VIEWS


class RelationGate(nn.Module):
    """Native R1 priors plus optional task and per-person log-odds corrections."""

    def __init__(self, mode):
        super().__init__()
        if mode not in ('fixed', 'static', 'person'):
            raise ValueError(f'Unknown relation gate: {mode}')
        self.mode = mode
        prior = torch.tensor([math.log(.16 / .64), math.log(.20 / .64)])
        self.register_buffer('prior', prior)
        if mode != 'fixed':
            self.offset = nn.Parameter(torch.zeros(2))
        if mode == 'person':
            self.scorers = nn.ModuleDict()
            for view in ('person', 'face'):
                scorer = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1, bias=False))
                nn.init.zeros_(scorer[-1].weight)
                self.scorers[view] = scorer

    def forward(self, reliable, person_relation=None, face_relation=None):
        with torch.autocast(device_type=reliable.device.type, enabled=False):
            reliable = reliable.reshape(-1).bool()
            odds = self.prior if self.mode == 'fixed' else self.prior + self.offset
            odds = odds.expand(len(reliable), -1)
            if self.mode == 'person':
                # Remove missing Face evidence before its scorer as well as at softmax.
                face_relation = torch.where(reliable[:, None], face_relation, 0.0)
                correction = torch.cat((self.scorers['person'](person_relation.float()),
                                        self.scorers['face'](face_relation.float())), dim=-1)
                odds = odds + correction
            full = odds.new_zeros(len(reliable))
            face = odds[:, 1].masked_fill(~reliable, -torch.inf)
            return torch.stack((full, odds[:, 0], face), dim=-1).softmax(dim=-1)


class D9RelationFusion(nn.Module):
    """Independent native Full and patch-receiving Person/Face, one readout per task."""

    def __init__(self, branches, mode):
        super().__init__()
        if set(branches) != set(VIEWS):
            raise ValueError('Joint D9 model requires Full, Person, and Face')
        self.mode = mode
        self.branches = nn.ModuleDict(branches)
        self.task_sizes = tuple(self.branches['full'].task_sizes)
        if any(tuple(model.task_sizes) != self.task_sizes for model in self.branches.values()):
            raise ValueError('View task partitions differ')
        # Constructing a scorer must not alter subsequent model or data RNG draws.
        with torch.random.fork_rng(devices=[]):
            self.gates = nn.ModuleList(RelationGate(mode) for _ in self.task_sizes)
        self._configure_gates(self.current_task_id)

    @property
    def current_task_id(self):
        return self.branches['full'].current_task_id

    def _configure_gates(self, task):
        for index, gate in enumerate(self.gates):
            gate.requires_grad_(index == task)

    def activate_task(self, task):
        for model in self.branches.values():
            model.activate_task(task)
        self._configure_gates(task)

    def restore_task(self, task):
        for model in self.branches.values():
            model.restore_task(task)
        self._configure_gates(task)

    def forward(self, inputs, *, all_seen=False):
        current = self.current_task_id
        if current < 0:
            raise RuntimeError('Activate or restore a task before inference')
        lanes = list(range(current + 1)) if all_seen else [current]
        reliable = inputs['evidence_face_reliable'].reshape(-1).bool()
        lane_logits, relations = {}, {'person': {}, 'face': {}}
        # The same batch's deterministic Full sender passes through the same
        # frozen CLIP in both receivers. Share only raw detached sender patches.
        shared_tower = self.branches['person'].visual_encoder is self.branches['face'].visual_encoder
        full_sender_cache = {} if shared_tower else None
        for view in VIEWS:
            branch = self.branches[view]
            if view == 'full':
                lane_logits[view] = branch.lane_logits(inputs['full'], all_seen)
            else:
                # Receivers use "full" as the historical key for their own pixels.
                own_inputs = {**inputs, 'full': inputs[view]}
                capture = relations[view] if self.mode == 'person' else None
                features = branch.encode_lanes(own_inputs, all_seen, full_relations=capture,
                                               full_sender_cache=full_sender_cache)
                lane_logits[view] = branch.head(features)

        weights, fused = [], []
        outputs = {view: [] for view in VIEWS}
        for lane, task in enumerate(lanes):
            start = sum(self.task_sizes[:task])
            columns = slice(start, start + self.task_sizes[task])
            for view in VIEWS:
                outputs[view].append(lane_logits[view][:, lane, columns])
            w = self.gates[task](reliable, relations['person'].get(task), relations['face'].get(task))
            scores = torch.stack([outputs[view][-1].float() for view in VIEWS], dim=1)
            # Unreliable Face may be a dummy crop. Exclude even a non-finite output.
            face = torch.where(reliable[:, None], scores[:, 2], 0.0)
            fused.append(w[:, 0:1] * scores[:, 0] + w[:, 1:2] * scores[:, 1] + w[:, 2:3] * face)
            weights.append(w)
        return dict(fused=torch.cat(fused, dim=1), weights=torch.stack(weights, dim=1),
                    branches={v: torch.cat(parts, dim=1) for v, parts in outputs.items()},
                    all_columns={v: scores[:, -1] for v, scores in lane_logits.items()})
