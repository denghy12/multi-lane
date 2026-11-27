"""Learn task-private logit weights from patch correspondences at the frozen receiver site."""

import math

import torch
from torch import nn

from .r1_crossvit import VIEWS


READOUTS = ('fixed', 'static', 'relation', 'summary')
DIRECTION_READOUTS = ('fixed', 'center', 'a', 'b', 'c', 'center_b')


def view_weights(odds, reliable):
    full = odds.new_zeros(len(reliable))
    face = odds[:, 1].masked_fill(~reliable, -torch.inf)
    return torch.stack((full, odds[:, 0], face), dim=-1).softmax(dim=-1)


def fuse_view_logits(scores, weights, reliable):
    # A missing Face can have a placeholder score. Exclude even a non-finite value.
    face = torch.where(reliable[:, None], scores[:, 2], 0.0)
    return (weights[:, 0:1] * scores[:, 0] + weights[:, 1:2] * scores[:, 1]
            + weights[:, 2:3] * face)


SAMPLE_GATES = {
    'sample_full': ('mid_bce', 'late_bce'),
    'sample_holdout': ('mid_bce', 'late_bce', 'mid_asl'),
    'sample_full_asl': ('mid_asl',),
}
SAMPLE_MODES = tuple(SAMPLE_GATES)


FILL_LEGACY_GATES = ('global_relation', 'task_summary', 'bounded_bce', 'own_benefit', 'shuffled_benefit', 'shared_static')
FILL_GATES = (*FILL_LEGACY_GATES, 'final_cosine')


def sample_readouts(mode, extra=()):
    if mode not in SAMPLE_MODES:
        raise ValueError(f'Unknown sample-level protocol: {mode}')
    return ('fixed', 'equal', *SAMPLE_GATES[mode], *extra)


class SampleFusionGate(nn.Module):
    """Learn unbiased per-person weights over exactly the available views."""

    def __init__(self, cls_width=None):
        super().__init__()
        self.linear = nn.Parameter(torch.zeros(3, 8))
        self.group_bias = nn.Parameter(torch.zeros(2, 3))
        self.late = cls_width is not None
        if self.late:
            # Eight fixed projection coordinates, no TRAIN/Val/Test fitting.
            # Unit-norm CLS times a Gaussian column has order-one scale.
            generator = torch.Generator().manual_seed(1)
            for view, dims in zip(VIEWS, (3, 3, 2)):
                self.register_buffer(view + '_projection', torch.randn(cls_width, dims, generator=generator))

    def forward(self, reliable, feature_by_view):
        with torch.autocast(device_type=reliable.device.type, enabled=False):
            parts = []
            for view in VIEWS if self.late else ('person', 'face'):
                values = feature_by_view[view].detach().float()
                if view == 'face':
                    values = torch.where(reliable[:, None], values, 0.)
                if self.late:
                    values = values @ getattr(self, view + '_projection')
                parts.append(values)
            scores = torch.cat(parts, dim=-1) @ self.linear.T
            scores = scores + torch.where(reliable[:, None], self.group_bias[1], self.group_bias[0])
            face = scores[:, 2].masked_fill(~reliable, -torch.inf)
            return torch.stack((scores[:, 0], scores[:, 1], face), dim=-1).softmax(dim=-1)


class FinalSimilarityGate(nn.Module):
    """Three explicit final-CLS cosine similarities; no learnt feature projection."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Parameter(torch.zeros(3, 3))
        self.group_bias = nn.Parameter(torch.zeros(2, 3))

    @staticmethod
    def features(reliable, features):
        values = {v: torch.nn.functional.normalize(features[v].detach().float(), dim=-1)
                  for v in VIEWS}
        cosine = torch.stack([(values[a] * values[b]).sum(-1)
                              for a, b in (('full', 'person'), ('full', 'face'), ('person', 'face'))], -1)
        return torch.cat((cosine[:, :1], torch.where(reliable[:, None], cosine[:, 1:], 0.)), -1)

    def from_features(self, reliable, cosine):
        with torch.autocast(device_type=reliable.device.type, enabled=False):
            scores = cosine.detach().float() @ self.linear.T + self.group_bias[reliable.long()]
            scores = torch.cat((scores[:, :2], scores[:, 2:].masked_fill(~reliable[:, None], -torch.inf)), -1)
            return scores.softmax(-1)

    def forward(self, reliable, features):
        return self.from_features(reliable, self.features(reliable, features))


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
            return view_weights(odds, reliable)


class SummaryGate(nn.Module):
    """Predict per-person view log-odds from the three actual Task10 summaries."""

    def __init__(self, width):
        super().__init__()
        hidden = 8
        self.projections = nn.ModuleDict({view: nn.Linear(width, hidden) for view in VIEWS})
        self.readout = nn.Sequential(nn.Linear(len(VIEWS) * hidden + 1, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 2))
        self.register_buffer('prior', torch.tensor([math.log(.16 / .64), math.log(.20 / .64)]))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, reliable, summaries):
        with torch.autocast(device_type=reliable.device.type, enabled=False):
            projected = []
            for view in VIEWS:
                value = summaries[view].detach().float()
                if view == 'face':
                    value = torch.where(reliable[:, None], value, 0.0)
                feature = self.projections[view](value).tanh()
                if view == 'face':
                    feature = torch.where(reliable[:, None], feature, 0.0)
                projected.append(feature)
            descriptor = torch.cat([*projected, reliable[:, None].float()], dim=1)
            return view_weights(self.prior + self.readout(descriptor), reliable)


class LinearDirectionGate(nn.Module):
    """Read detached Full-association statistics into bounded auxiliary log-odds."""

    def __init__(self):
        super().__init__()
        self.register_buffer('prior', torch.tensor([math.log(.16 / .64), math.log(.20 / .64)]))
        self.scorers = nn.ModuleDict({view: nn.Linear(4, 1, bias=False)
                                     for view in ('person', 'face')})
        for scorer in self.scorers.values():
            nn.init.zeros_(scorer.weight)

    def forward(self, reliable, person_relation, face_relation, *, base_odds=None):
        with torch.autocast(device_type=reliable.device.type, enabled=False):
            reliable = reliable.reshape(-1).bool()
            person = person_relation.detach().float()
            face = torch.where(reliable[:, None], face_relation.detach().float(), 0.0)
            raw = torch.cat((self.scorers['person'](person), self.scorers['face'](face)), dim=-1)
            center = self.prior if base_odds is None else base_odds.detach().float()
            return view_weights(center + raw.tanh(), reliable), raw


class D9RelationFusion(nn.Module):
    """Independent native Full and patch-receiving Person/Face, one readout per task."""

    def __init__(self, branches, mode, extra_readouts=()):
        super().__init__()
        if set(branches) != set(VIEWS):
            raise ValueError('Joint D9 model requires Full, Person, and Face')
        self.mode = mode
        self.extra_readouts = tuple(extra_readouts)
        if len(set(self.extra_readouts)) != len(self.extra_readouts) or set(self.extra_readouts) - set(FILL_GATES):
            raise ValueError('Unknown or repeated fill-table readout')
        if self.extra_readouts and mode not in SAMPLE_MODES:
            raise ValueError('Fill-table readouts require protected sample training')
        self.branches = nn.ModuleDict(branches)
        self.task_sizes = tuple(self.branches['full'].task_sizes)
        if any(tuple(model.task_sizes) != self.task_sizes for model in self.branches.values()):
            raise ValueError('View task partitions differ')
        # Constructing a scorer must not alter subsequent model or data RNG draws.
        with torch.random.fork_rng(devices=[]):
            if mode in SAMPLE_MODES:
                if any(self.branches[v].r1_layer != 8 for v in ('person', 'face')):
                    raise ValueError('Sample fusion requires Frozen9 patch receivers')
                self.gates = nn.ModuleList(nn.ModuleDict({
                    'fixed': RelationGate('fixed'),
                    **{name: SampleFusionGate(self.branches['full'].output_dim if name == 'late_bce' else None)
                       for name in SAMPLE_GATES[mode]},
                }) for _ in self.task_sizes)
            elif mode == 'multi':
                # Draw all relation scorers first to replay the original relation initialization.
                relations = [RelationGate('person') for _ in self.task_sizes]
                if any(self.branches[v].r1_layer != 8 for v in ('person', 'face')):
                    raise ValueError('Protected multi readouts require both patch receivers at block 9')
                if any(len(branch.visual_encoder.transformer.resblocks) < 10 for branch in branches.values()):
                    raise ValueError('Protected multi readouts require a tenth Task Forward block')
                self.gates = nn.ModuleList(nn.ModuleDict({
                    'fixed': RelationGate('fixed'), 'static': RelationGate('static'),
                    'relation': relation, 'summary': SummaryGate(self.branches['full'].width),
                }) for relation in relations)
            elif mode == 'direction':
                if any(self.branches[v].r1_layer != 8 for v in ('person', 'face')):
                    raise ValueError('Direction readouts require both patch receivers at block 9')
                self.gates = nn.ModuleList(nn.ModuleDict({
                    'fixed': RelationGate('fixed'), 'center': RelationGate('static'),
                    'a': LinearDirectionGate(), 'b': LinearDirectionGate(), 'c': LinearDirectionGate(),
                    'center_b': LinearDirectionGate(),
                }) for _ in self.task_sizes)
                for bank in self.gates:
                    bank['center'].prior.zero_()
                    bank['center_b'].prior.zero_()
            else:
                self.gates = nn.ModuleList(RelationGate(mode) for _ in self.task_sizes)
        if self.extra_readouts:
            # Isolate auxiliary initialization from all branch/data random streams.
            # Global-relation and summary ordering reproduces historical multi initialization.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(1).get_state())
                relations = [RelationGate('person') for _ in self.task_sizes]
                summaries = [SummaryGate(self.branches['full'].width) for _ in self.task_sizes]
                for task, bank in enumerate(self.gates):
                    constructors = {
                        'global_relation': lambda task=task: relations[task],
                        'task_summary': lambda task=task: summaries[task],
                        'shared_static': lambda: RelationGate('static'),
                        'bounded_bce': LinearDirectionGate,
                        'own_benefit': LinearDirectionGate,
                        'shuffled_benefit': LinearDirectionGate,
                        'final_cosine': FinalSimilarityGate,
                    }
                    for name in self.extra_readouts:
                        bank[name] = constructors[name]()
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

    def forward(self, inputs, *, all_seen=False, sample_gate_only=False, sample_branch_only=False,
                task10_probe=None, summary_tokens=None, final_tokens=None):
        current = self.current_task_id
        if current < 0:
            raise RuntimeError('Activate or restore a task before inference')
        lanes = list(range(current + 1)) if all_seen else [current]
        reliable = inputs['evidence_face_reliable'].reshape(-1).bool()
        lane_logits, relations = {}, {'person': {}, 'face': {}}
        multi = self.mode == 'multi'
        direction = self.mode == 'direction'
        sample_mode = self.mode in SAMPLE_MODES
        if (sample_gate_only or sample_branch_only) and not sample_mode:
            raise ValueError('Selective sample forward requires a sample-level training protocol')
        if sample_gate_only and sample_branch_only:
            raise ValueError('Choose either the branch or the gate training forward')
        sample = sample_mode and not sample_branch_only
        late_features = {}
        protected_readouts = multi or direction or sample
        need_summary = multi or (sample and 'task_summary' in self.extra_readouts)
        pooled = {v: {} for v in VIEWS} if need_summary else None
        global_relations = {v: {} for v in ('person', 'face')} if sample and self.extra_readouts else None
        # The same batch's deterministic Full sender passes through the same
        # frozen CLIP in both receivers. Share only raw detached sender patches.
        shared_tower = self.branches['person'].visual_encoder is self.branches['face'].visual_encoder
        full_sender_cache = {} if shared_tower else None
        with torch.set_grad_enabled(torch.is_grad_enabled() and not sample_gate_only):
            for view in VIEWS:
                branch = self.branches[view]
                summary_args = dict(summary_layer=9, pooled_summaries=pooled[view]) if need_summary else {}
                if final_tokens is not None:
                    if self.training or torch.is_grad_enabled() or not sample:
                        raise ValueError('Final token capture requires frozen sample-fusion inference')
                    final_tokens[view] = {}
                    summary_args['final_tokens'] = final_tokens[view]
                if summary_tokens is not None:
                    if self.training or torch.is_grad_enabled() or not sample:
                        raise ValueError('Summary capture requires frozen sample-fusion inference')
                    summary_tokens[view] = {}
                    summary_args.update(summary_layer=9, summary_tokens=summary_tokens[view])
                if task10_probe is not None:
                    if not sample or self.training or torch.is_grad_enabled():
                        raise ValueError('Task10 diagnostics require frozen sample-fusion inference')
                    task10_probe[view] = {}
                    summary_args['task10_probe'] = task10_probe[view]
                if view == 'full':
                    if multi or sample:
                        features = branch.encode_lanes(inputs['full'], all_seen, **summary_args)
                        lane_logits[view] = branch.head(features)
                    else:
                        lane_logits[view] = branch.lane_logits(inputs['full'], all_seen)
                else:
                    # Receivers use "full" as the historical key for their own pixels.
                    own_inputs = {**inputs, 'full': inputs[view]}
                    capture = relations[view] if protected_readouts or self.mode == 'person' else None
                    features = branch.encode_lanes(own_inputs, all_seen, full_relations=capture,
                                                   full_sender_cache=full_sender_cache,
                                                   detach_relation=protected_readouts,
                                                   selector_weight_relation=sample,
                                                   **({'global_full_relations': global_relations[view]} if global_relations is not None else {}),
                                                   **summary_args)
                    lane_logits[view] = branch.head(features)
                if sample:
                    late_features[view] = features.detach()

        names = {
            'multi': READOUTS,
            'direction': DIRECTION_READOUTS,
        }.get(self.mode, (self.mode,))
        if sample_mode:
            names = ('fixed',) if sample_branch_only else sample_readouts(self.mode, self.extra_readouts)
        all_weights = {name: [] for name in names}
        all_fused = {name: [] for name in names}
        direction_scores = {name: [] for name in ('a', 'b', 'c', 'center_b')} if direction else {}
        if sample:
            direction_scores.update({name: [] for name in self.extra_readouts
                                     if name in ('bounded_bce', 'own_benefit', 'shuffled_benefit')})
        outputs = {view: [] for view in VIEWS}
        for lane, task in enumerate(lanes):
            if sample:
                mid = {v: relations[v][task] for v in ('person', 'face')}
                late = {v: late_features[v][:, lane] for v in VIEWS}
                if task10_probe is not None and task == current:
                    task10_probe['relation'] = torch.cat((mid['person'], mid['face']), dim=1)
            start = sum(self.task_sizes[:task])
            columns = slice(start, start + self.task_sizes[task])
            for view in VIEWS:
                outputs[view].append(lane_logits[view][:, lane, columns])
            scores = torch.stack([outputs[view][-1].float() for view in VIEWS], dim=1)
            for name in names:
                if name == 'equal':
                    w = view_weights(scores.new_zeros(len(reliable), 2), reliable)
                elif name in ('mid_bce', 'mid_asl', 'late_bce'):
                    w = self.gates[task][name](reliable, late if name == 'late_bce' else mid)
                elif name in self.extra_readouts:
                    gate = self.gates[task][name]
                    if name == 'final_cosine':
                        w = gate(reliable, late)
                    elif name == 'task_summary':
                        w = gate(reliable, {v: pooled[v][task] for v in VIEWS})
                    elif name == 'shared_static':
                        w = gate(reliable)
                    elif name == 'global_relation':
                        w = gate(reliable, global_relations['person'][task], global_relations['face'][task])
                    else:
                        w, raw = gate(reliable, global_relations['person'][task], global_relations['face'][task])
                        direction_scores[name].append(raw)
                elif name == 'center_b':
                    center = self.gates[task]['center']
                    gate = self.gates[task][name]
                    w, raw = gate(reliable, relations['person'][task], relations['face'][task],
                                  base_odds=center.prior + center.offset)
                    direction_scores[name].append(raw)
                elif direction and name in ('a', 'b', 'c'):
                    gate = self.gates[task][name]
                    w, raw = gate(reliable, relations['person'][task], relations['face'][task])
                    direction_scores[name].append(raw)
                elif name == 'summary':
                    gate = self.gates[task][name]
                    w = gate(reliable, {v: pooled[v][task] for v in VIEWS})
                elif name == 'relation':
                    gate = self.gates[task][name]
                    w = gate(reliable, relations['person'][task].detach(), relations['face'][task].detach())
                else:
                    gate = self.gates[task][name] if protected_readouts or sample_mode else self.gates[task]
                    w = gate(reliable, relations['person'].get(task), relations['face'].get(task))
                readout_scores = scores.detach() if protected_readouts and name != 'fixed' else scores
                all_fused[name].append(fuse_view_logits(readout_scores, w, reliable))
                all_weights[name].append(w)

        readouts = {name: dict(fused=torch.cat(all_fused[name], dim=1),
                               weights=torch.stack(all_weights[name], dim=1)) for name in names}
        for name, raw in direction_scores.items():
            readouts[name]['direction_scores'] = torch.stack(raw, dim=1)
        result = dict(**readouts[names[0]],
                      branches={v: torch.cat(parts, dim=1) for v, parts in outputs.items()},
                      all_columns={v: scores[:, -1] for v, scores in lane_logits.items()})
        if protected_readouts or sample_mode:
            result['readouts'] = readouts
        if sample and self.extra_readouts and not all_seen:
            result['fill_features'] = {
                'global': {v: global_relations[v][current].detach() for v in ('person', 'face')},
                'cosine': FinalSimilarityGate.features(reliable, late),
                **({'summary': {v: pooled[v][current].detach() for v in VIEWS}} if need_summary else {}),
            }
        return result
