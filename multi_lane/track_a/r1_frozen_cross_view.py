"""Task-private cross-view receivers inside the frozen visual token stream.

Only the receiving stream retains input gradients. The CLIP weights and the
two sending streams remain frozen. Old task modules never change.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

from .r1_crossvit import CLSAgent, R1Receiver, VIEWS, restore_rng

MODES = ('image_cls', 'patch')


class FrozenCrossAgent(nn.Module):
    def __init__(self, width, dim=64, heads=4, mode='image_cls', cap=.05):
        super().__init__()
        if mode not in MODES or dim % heads or not 0 < cap <= 1:
            raise ValueError('Invalid frozen cross-view configuration')
        self.mode, self.dim, self.heads, self.cap = mode, dim, heads, cap
        self.types = nn.Parameter(torch.randn(2, width) * .02)
        self.q = nn.Linear(width, dim, bias=False)
        self.k = nn.Linear(width, dim, bias=False)
        self.v = nn.Linear(width, dim, bias=False)
        self.out = nn.Linear(dim, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.diagnostic_enabled = False
        self.last_diagnostic = None

    def forward(self, tokens, memories, masks):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            own = tokens.float()
            cls = F.layer_norm(own[:, :1], (own.shape[-1],))
            query = cls if self.mode == 'image_cls' else F.layer_norm(own[:, 1:], (own.shape[-1],))
            sources = [F.layer_norm(m.detach().float() + self.types[i], (m.shape[-1],))
                       for i, m in enumerate(memories)]
            memory = torch.cat([cls, *sources], 1)
            batch, head_width = len(own), self.dim // self.heads
            def split(value):
                return value.reshape(batch, -1, self.heads, head_width).transpose(1, 2)
            raw_q, raw_k = split(self.q(query)), split(self.k(memory))
            q = F.layer_norm(raw_q, (head_width,))
            k = F.layer_norm(raw_k, (head_width,))
            v = split(self.v(memory))
            valid = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=own.device),
                *[mask[:, None].expand(-1, src.shape[1]) for mask, src in zip(masks, sources)]], 1)
            scores = (q @ k.transpose(-2, -1)) * head_width ** -.5
            attention = scores.masked_fill(~valid[:, None, None], float('-inf')).softmax(-1)
            context = (attention @ v).transpose(1, 2).reshape(batch, -1, self.dim)
            raw = self.out(context)
            reference = own[:, :1].detach().norm(dim=(1, 2)).clamp_min(1e-8)
            raw_norm = raw.norm(dim=(1, 2))
            gain = (self.cap * reference / raw_norm.clamp_min(1e-8)).clamp(max=1)
            bounded = raw * gain[:, None, None]
            delta = (torch.cat([bounded, torch.zeros_like(own[:, 1:])], 1)
                     if self.mode == 'image_cls' else torch.cat([torch.zeros_like(own[:, :1]), bounded], 1))
            if self.diagnostic_enabled:
                with torch.no_grad():
                    masses, offset = [], 0
                    for size in [1, *[s.shape[1] for s in sources]]:
                        masses.append(float(attention[..., offset:offset+size].sum(-1).mean()))
                        offset += size
                    token_ratio = delta.norm(dim=-1) / own.detach().norm(dim=-1).clamp_min(1e-8)
                    self.last_diagnostic = dict(
                        query_count=query.shape[1], memory_count=memory.shape[1], source_masses=masses,
                        projected_q_norm=float(raw_q.norm(dim=-1).mean()),
                        projected_k_norm=float(raw_k.norm(dim=-1).mean()),
                        normalized_q_norm=float(q.norm(dim=-1).mean()),
                        normalized_k_norm=float(k.norm(dim=-1).mean()),
                        score_range=float((scores.max(-1).values-scores.min(-1).values).mean()),
                        attention_entropy=float(-(attention*attention.clamp_min(1e-30).log()).sum(-1).mean()),
                        residual_to_reference=float((bounded.norm(dim=(1, 2))/reference).mean()),
                        cap_hit_fraction=float((gain < 1).float().mean()),
                        sequence_relative_change=float((delta.norm(dim=(1, 2))/own.detach().norm(dim=(1, 2)).clamp_min(1e-8)).mean()),
                        max_token_relative_change=float(token_ratio.max()),
                        mean_token_relative_change=float(token_ratio.mean()))
        return tokens + delta.to(tokens.dtype)


class FrozenR1Receiver(R1Receiver):
    def __init__(self, *args, frozen_cross_mode='image_cls', frozen_cross_cap=.05,
                 frozen_cross_second_layer=None, frozen_cross_sharing='shared',
                 task_pool_layer=None, recovery_source=None, **kwargs):
        if kwargs.get('r1_view') not in ('person', 'face'):
            raise ValueError('Full is protected; only Person and Face may receive')
        if kwargs.get('r1_sender_root') is not None or kwargs.get('r1_query_mode', 'cls') != 'cls':
            raise ValueError('Frozen receivers only read frozen spatial evidence')
        kwargs['r1_kind'] = 'native'
        super().__init__(*args, **kwargs)
        if self.r1_layer >= len(self.visual_encoder.transformer.resblocks) - 1:
            raise ValueError('A later Summarize block is required after visual injection')
        if frozen_cross_sharing not in ('shared', 'independent'):
            raise ValueError('Unknown cross-view parameter sharing')
        if frozen_cross_second_layer is not None:
            if (frozen_cross_mode != 'patch' or not self.r1_layer < frozen_cross_second_layer
                    < len(self.visual_encoder.transformer.resblocks) - 1):
                raise ValueError('Two distinct ordered patch sites must precede a later Summarize block')
        elif frozen_cross_sharing == 'independent':
            raise ValueError('Independent site parameters require two sites')
        if task_pool_layer is not None and (
                self.r1_view != 'face' or frozen_cross_mode != 'patch'
                or frozen_cross_second_layer is not None
                or not 0 <= task_pool_layer < self.r1_layer):
            raise ValueError('Task Pool requires Face, one later frozen patch site, and ordered layers')
        self.frozen_cross_mode, self.frozen_cross_cap = frozen_cross_mode, frozen_cross_cap
        self.frozen_cross_layers = ((self.r1_layer,) if frozen_cross_second_layer is None
                                    else (self.r1_layer, frozen_cross_second_layer))
        self.frozen_cross_sharing = frozen_cross_sharing
        self.frozen_cross_active_layers = self.frozen_cross_layers
        self.task_pool_layer = task_pool_layer
        self.task_pool_enabled = task_pool_layer is not None
        self.r1_kind = 'attention'
        self.recovery_source = Path(recovery_source) if recovery_source else None
        def new_agent():
            return FrozenCrossAgent(self.width, kwargs.get('r1_dim', 64), kwargs.get('r1_heads', 4),
                                    frozen_cross_mode, frozen_cross_cap / len(self.frozen_cross_layers))
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(20261105)
            first = [new_agent() for _ in self.task_sizes]
            if frozen_cross_sharing == 'shared':
                self.r1_bank.extend(first)
            else:
                self.r1_bank.extend(nn.ModuleList((agent, new_agent())) for agent in first)
        if task_pool_layer is not None:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(20261102)
                self.r1_bank = nn.ModuleList(
                    nn.ModuleList((agent, CLSAgent(self.width, kwargs.get('r1_dim', 64),
                                                  kwargs.get('r1_heads', 4), pool=True)))
                    for agent in self.r1_bank
                )

    def cross_agent(self, task, layer):
        task_bank = self.r1_bank[task]
        if self.task_pool_layer is not None:
            return task_bank[0]
        if self.frozen_cross_sharing == 'shared':
            return task_bank
        return task_bank[self.frozen_cross_layers.index(layer)]

    def _load_external(self, path, task, training=False):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['task'] != task:
            raise ValueError('External recovery task differs')
        original, current = saved['config'], self.r1_config
        allowed = {'git', 'r1_fixed_loss_scale', 'r1_loss_scale_growth_interval', 'r1_recovery_source'}
        changed = [k for k in original.keys() | current.keys()
                   if k not in allowed and original.get(k) != current.get(k)]
        if changed:
            raise ValueError(f'External recovery recipe changed: {changed}')
        history = saved['training']['history'] if training else saved['history']
        if any(r['skipped_optimizer_steps'] for r in history):
            raise ValueError('External recovery contains skipped updates')
        if training and (saved['training']['next_epoch'] != 18
                         or saved['training']['completed_task_updates'] != 2808):
            raise ValueError('Only the complete task3 epoch18 boundary is allowed')
        missing = self.load_state_dict(saved['model'], strict=False)
        if missing.unexpected_keys or any(not n.startswith('visual_encoder.') for n in missing.missing_keys):
            raise ValueError('Incomplete external state')
        self.restore_task(task)
        if training:
            from .runner import build_optimizer_groups
            _, _, groups = build_optimizer_groups(self, 0., adapter_learning_rate=current['adapter_learning_rate'])
            names = {id(p):n for n,p in self.named_parameters()}
            actual = [[names[id(p)] for p in group['params']] for group in groups]
            if actual != saved['training']['optimizer_parameter_names']:
                raise ValueError('Optimizer parameter ordering changed')
            self.r1_resume = saved['training']
        restore_rng(saved['rng'])
        if self.r1_root:
            from .r1_crossvit_experiment import write
            write(self.r1_root/f'external_task{task}_provenance.json', dict(
                source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_config=original, numerical_policy_only_change=True, epoch18=training))
        return saved

    def restore_completed_task(self, task):
        own = super().restore_completed_task(task)
        if own is not None:
            return own
        if self.recovery_source is not None and task < 3:
            return self._load_external(self.recovery_source/'task_recovery'/f'task{task}.pth', task)
        return None

    def activate_task(self, task):
        super().activate_task(task)
        if self.recovery_source is not None and task == 3 and self.r1_resume is None:
            self._load_external(self.recovery_source/'epoch_recovery/task3/epoch18.pth', task, True)

    def save_training_epoch(self, task, epoch, history, optimizer, scheduler, scaler, updates):
        history[-1]['cross_view_last_gradient_norms'] = {
            n: None if p.grad is None else float(p.grad.float().norm())
            for n, p in self.r1_bank[task].named_parameters()}
        if torch.cuda.is_available():
            history[-1]['cuda_peak_allocated_mib'] = torch.cuda.max_memory_allocated()/1024**2
            history[-1]['cuda_peak_reserved_mib'] = torch.cuda.max_memory_reserved()/1024**2
        super().save_training_epoch(task, epoch, history, optimizer, scheduler, scaler, updates)

    def encode_lanes(self, images, all_seen_lanes):
        own = images['full'] if isinstance(images, dict) else images
        if not self.r1_enabled:
            return super().encode_lanes(own, all_seen_lanes)
        if not isinstance(images, dict):
            raise ValueError('Cross-view evidence is required')
        lanes = self._lane_ids(all_seen_lanes)
        blocks = list(self.visual_encoder.transformer.resblocks)
        sites = self.frozen_cross_active_layers
        pool_site = self.task_pool_layer if self.task_pool_enabled else None
        evidence_sites = set(sites)
        if pool_site is not None:
            evidence_sites.add(pool_site)
        if not evidence_sites or any(site not in self.frozen_cross_layers for site in sites):
            raise ValueError('Use the receiver switch to disable all sites')
        memories = {site: [] for site in evidence_sites}
        masks = []
        with torch.no_grad():
            for view in VIEWS:
                if view == self.r1_view:
                    continue
                source = self._visual_tokens(images['evidence_' + view])
                for layer, block in enumerate(blocks[:max(evidence_sites)+1]):
                    source = block(source.permute(1, 0, 2)).permute(1, 0, 2)
                    if layer in memories:
                        memories[layer].append(source[:, 1:].detach())
                masks.append(images['evidence_face_reliable'].bool() if view == 'face'
                             else torch.ones(len(own), dtype=torch.bool, device=own.device))
            tokens = self._visual_tokens(own)
        self.adapter_bank.reset_auxiliary_metrics()
        state = self._initial_lane_tokens(len(own), lanes)
        prefix_end = self.r1_layer if sites else len(blocks) - 1
        for layer, block in enumerate(blocks[:prefix_end+1]):
            state = self._lane_block(block, tokens, state, lanes, layer)
            with torch.no_grad():
                tokens = block(tokens.permute(1, 0, 2)).permute(1, 0, 2)
            if layer == pool_site:
                state = torch.stack([
                    torch.cat((self.r1_bank[task][1](state[i, :, :1], memories[layer], masks),
                               state[i, :, 1:]), dim=1)
                    for i, task in enumerate(lanes)
                ])
        # `tokens` is a shared read-only prefix, never overwritten in place.
        outputs = []
        for i, task in enumerate(lanes):
            stream = tokens
            if self.r1_layer in sites:
                stream = self.cross_agent(task, self.r1_layer)(stream, memories[self.r1_layer], masks)
            lane = state[i:i+1]
            for layer in range(prefix_end+1, len(blocks)):
                block = blocks[layer]
                lane = self._lane_block(block, stream, lane, [task], layer,
                                        preserve_image_gradient=True)
                # The last visual block has no later Task Forward reader.
                if len(self.frozen_cross_layers) == 1 or layer < len(blocks) - 1:
                    stream = block(stream.permute(1, 0, 2)).permute(1, 0, 2)
                if layer in sites:
                    stream = self.cross_agent(task, layer)(stream, memories[layer], masks)
            outputs.append(lane)
        state = self.visual_encoder.ln_post(torch.cat(outputs, 0))
        if self.visual_encoder.proj is not None:
            state = state @ self.visual_encoder.proj
        return F.normalize(state[:, :, 0].permute(1, 0, 2).float(), dim=-1)
