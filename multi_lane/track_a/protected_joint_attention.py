"""Private task-local interaction readout; the original three lanes stay intact.

No teacher, pretrained task tensor, second classifier, or source model is used.
Only this branch's loss is detached at the base representation/head boundary.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import MultiLaneModel

ARMS = ('C0', 'C1', 'C2', 'C3', 'C4', 'FULL_ONLY')
VIEWS = ('full', 'person', 'face')


def bounded_delta(delta, reference, ratio=.1):
    base_sq = reference.detach().float().square().mean(-1, keepdim=True)
    delta_sq = delta.float().square().mean(-1, keepdim=True)
    factor = (1 + delta_sq / (ratio**2 * base_sq).clamp_min(1e-20)).rsqrt()
    return torch.where(base_sq > 0, delta.float() * factor, torch.zeros_like(delta.float()))


def target_summary(tokens, selectors, boxes, visibility):
    """Hard patch-overlap mask, without the global CLS or all-masked softmax."""
    count = tokens.shape[-2] - 1
    grid = math.isqrt(count)
    if grid**2 != count or boxes.shape != (tokens.shape[0], 4):
        raise ValueError('Expected a square patch grid and [batch,4] mapped boxes')
    axis = torch.arange(grid, device=tokens.device, dtype=torch.float32) / grid
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    x, y = x.flatten()[None], y.flatten()[None]
    boxes = boxes.detach().float()
    valid = torch.isfinite(boxes).all(-1) & torch.isfinite(visibility)
    valid &= (visibility > 0) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    overlap = ((x + 1/grid > boxes[:, 0, None]) & (x < boxes[:, 2, None])
               & (y + 1/grid > boxes[:, 1, None]) & (y < boxes[:, 3, None]))
    overlap &= valid[:, None]
    valid &= overlap.any(-1)
    # Invisible targets get a safe temporary distribution and a zero output.
    safe_mask = overlap | ~valid[:, None]
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        patches, query = tokens.detach().float()[:, 1:], selectors.detach().float()
        logits = torch.einsum('tbsc,bnc->tbsn', query, patches) * tokens.shape[-1]**-.5
        weights = logits.masked_fill(~safe_mask[None, :, None], float('-inf')).softmax(-1)
        summary = torch.einsum('tbsn,bnc->tbsc', weights, patches)
        summary = torch.where(valid[None, :, None, None], summary, torch.zeros_like(summary))
    return summary, valid


class InteractionReadout(nn.Module):
    def __init__(self, width, output_dim, arm, dim=64, heads=4):
        super().__init__()
        if arm not in ARMS[1:] or dim % heads:
            raise ValueError('Invalid private interaction configuration')
        self.arm, self.dim, self.heads = arm, dim, heads
        self.output_dim = output_dim
        if arm == 'C1':
            # ~197k/task at CLIP dimensions, within 1% of C2's ~199k.
            self.mlp = nn.Sequential(nn.Linear(3*output_dim, 96), nn.GELU(),
                                     nn.Linear(96, output_dim))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.query = nn.Linear(2*output_dim, dim, bias=False)
            self.key = nn.Linear(width, dim, bias=False)
            self.value = nn.Linear(width, dim, bias=False)
            self.out = nn.Linear(dim, output_dim, bias=False)
            self.view_types = nn.Parameter(torch.empty(3, width))
            nn.init.normal_(self.view_types, std=.01)
            nn.init.zeros_(self.out.weight)
        self.gate = None
        if arm in ('C4', 'FULL_ONLY'):
            self.gate = nn.Sequential(nn.Linear(3*output_dim+2, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(.1/.9))
        self.last_metrics = {}

    def forward(self, base, features, summaries, reliable, full_visible, no_face=False, no_person=False):
        with torch.autocast(device_type=base.device.type, enabled=False):
            base = base.detach().float()
            f, p, face = [features[n].detach().float() for n in VIEWS]
            use_face = reliable.detach().bool() & (not no_face) & (self.arm != 'FULL_ONLY')
            face = torch.where(use_face[:, None], face, torch.zeros_like(face))
            if no_person or self.arm == 'FULL_ONLY':
                p = torch.zeros_like(p)
            gate_base = f if self.arm == 'FULL_ONLY' else base
            if self.arm == 'C1':
                raw = self.mlp(torch.cat([f, p, face], -1))
                face_mass = raw.new_zeros(len(raw))
            else:
                query = self.query(F.layer_norm(torch.cat([gate_base, p], -1), (2*self.output_dim,)))
                memories, masks = [], []
                for index, name in enumerate(VIEWS):
                    memory = summaries[name].detach().float()
                    allowed = (full_visible.bool() if name == 'full' else use_face if name == 'face'
                               else torch.full_like(use_face, not no_person and self.arm != 'FULL_ONLY'))
                    memory = torch.where(allowed[:, None, None], memory, torch.zeros_like(memory))
                    memory = F.layer_norm(memory, (memory.shape[-1],)) + self.view_types[index]
                    memories.append(memory)
                    masks.append(allowed[:, None].expand(-1, memory.shape[1]))
                memory = torch.cat(memories, 1)
                valid = torch.cat(masks, 1)
                k, v = self.key(memory), self.value(memory)
                # Always-valid zero-valued null evidence permits a no-read choice.
                k = F.pad(k, (0, 0, 0, 1)); v = F.pad(v, (0, 0, 0, 1))
                valid = F.pad(valid, (0, 1), value=True)
                b, n = valid.shape
                q = query.reshape(b, self.heads, 1, -1)
                k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
                v = v.reshape(b, n, self.heads, -1).transpose(1, 2)
                weights = ((q @ k.transpose(-1, -2)) / math.sqrt(self.dim/self.heads))
                weights = weights.masked_fill(~valid[:, None, None], float('-inf')).softmax(-1)
                raw = self.out((weights @ v).reshape(b, self.dim))
                start = sum(m.shape[1] for m in memories[:2])
                face_mass = weights[..., start:-1].sum(-1).mean((1, 2))
            gate = self._gate_value(base, gate_base, p, face, use_face, full_visible)
            delta = self._correction(raw, base, gate)
            self.last_metrics = dict(gate_mean=float(gate.detach().mean()),
                delta_to_base_rms=float((delta.detach().square().mean(-1).sqrt()
                    / base.square().mean(-1).sqrt().clamp_min(1e-12)).mean()),
                face_attention_mass=float(face_mass.detach().mean()))
            return delta

    def _gate_value(self, base, gate_base, person, face, use_face, full_visible):
        gate = base.new_full((len(base), 1), .1)
        if self.gate is not None:
            inputs = torch.cat([gate_base, person, face, use_face[:, None].float(),
                                full_visible[:, None].float()], -1)
            gate = self.gate(inputs).sigmoid()
        return gate

    def _correction(self, raw, base, gate):
        return gate * bounded_delta(raw, base)


class ProtectedJointModel(MultiLaneModel):
    def initialize_interaction(self, arm, layer=5, dim=64, heads=4):
        if arm not in ARMS or not 0 <= layer < len(self.visual_encoder.transformer.resblocks):
            raise ValueError('Invalid interaction arm/layer')
        self.interaction_arm, self.interaction_layer = arm, layer
        self.interaction_enabled = arm != 'C0'
        # CPU-only, isolated initialization cannot advance any base CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(928100)
            self.interaction_bank = nn.ModuleList([
                InteractionReadout(self.width, self.head.in_features,
                                   arm, dim, heads) for _ in range(self.num_tasks)
            ]) if arm != 'C0' else nn.ModuleList()
        self.interaction_bank.to(self.head.weight.device).requires_grad_(False)
        self._interaction_cache = {}

    def _lane_block(self, block, image_tokens, *args, **kwargs):
        self._interaction_image_source = (block, image_tokens)
        try:
            return super()._lane_block(block, image_tokens, *args, **kwargs)
        finally:
            self._interaction_image_source = None

    def _task_attention_ffn(self, block, lane_tokens, lane_ids, layer_id,
                            task_cls, selectors, selected, image_view='full', summary_valid=None):
        if self.interaction_bank and layer_id == self.interaction_layer:
            self._interaction_cache[image_view] = selected.detach()
            if image_view == 'full' and self.interaction_arm in ('C3', 'C4', 'FULL_ONLY'):
                image_block, image_tokens = self._interaction_image_source
                with torch.no_grad():
                    normalized = image_block.ln_1(image_tokens)
                    region, visible = target_summary(normalized, selectors,
                        self._interaction_inputs['spatial_bbox'],
                        self._interaction_inputs['spatial_visibility'])
                self._interaction_cache['full'] = region
                self._interaction_cache['visible'] = visible
        return super()._task_attention_ffn(block, lane_tokens, lane_ids, layer_id,
            task_cls, selectors, selected, image_view, summary_valid)

    def encode_lanes_with_views(self, images, all_seen_lanes):
        self._interaction_cache = {}
        self._interaction_inputs = images
        try:
            base, features = super().encode_lanes_with_views(images, all_seen_lanes)
        finally:
            self._interaction_inputs = None
        self._interaction_cache.update(base=base.detach(),
            features={k:v.detach() for k,v in features.items()},
            reliable=images['face_reliable'].detach().bool(),
            lane_ids=self._lane_ids(all_seen_lanes))
        # The original training forward never contains the new correction.
        if self.training or not self.interaction_enabled:
            return base, features
        return base + self.interaction_delta(), features

    def interaction_delta(self, no_face=False, no_person=False):
        cache = self._interaction_cache
        if not self.interaction_bank:
            return torch.zeros_like(cache['base'])
        b = cache['base'].shape[0]
        visible = cache.get('visible', torch.ones(b, dtype=torch.bool, device=cache['base'].device))
        rows = []
        for index, task in enumerate(cache['lane_ids']):
            summaries = {v:cache[v][index] for v in VIEWS} if self.interaction_arm != 'C1' else {}
            rows.append(self.interaction_bank[task](cache['base'][:, index],
                {v:cache['features'][v][:, index] for v in VIEWS}, summaries,
                cache['reliable'], visible, no_face, no_person))
        return torch.stack(rows, 1)

    def interaction_parameters(self):
        if self.interaction_bank and self.current_task_id >= 0:
            yield from self.interaction_bank[self.current_task_id].parameters()

    def configure_interaction(self):
        self.interaction_bank.requires_grad_(False)
        for p in self.interaction_bank.parameters():
            p.grad = None
        if self.interaction_bank and self.current_task_id >= 0:
            self.interaction_bank[self.current_task_id].requires_grad_(True)

    def cached_base_seen_logits(self, delta=None):
        base = self._interaction_cache['base']
        lane_ids = self._interaction_cache['lane_ids']
        logits = self._lane_logits_with_views(base if delta is None else base+delta,
            self._interaction_cache['features'], None, lane_ids)[0]
        return torch.cat([logits[:, i, sum(self._task_sizes[:t]):sum(self._task_sizes[:t+1])]
                          for i,t in enumerate(lane_ids)], -1)


class PrivateInteractionTraining:
    """An independent optimizer/scaler; no base/head gradient is installed."""
    def __init__(self, model, epochs=30, amp=True):
        self.model = model
        self.parameters = tuple(model.interaction_parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=.0004, weight_decay=0.)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=65536.)

    def begin_epoch(self):
        self.values, self.steps, self.skips = [], 0, 0

    def backward(self, targets, current):
        from .runner import compute_training_loss
        self.optimizer.zero_grad(set_to_none=True)
        delta = self.model.interaction_delta()[:, 0]
        fused = self.model._interaction_cache['base'][:, 0] + delta
        with torch.autocast(device_type=fused.device.type, enabled=False):
            logits = F.linear(fused.float(), self.model.head.weight.detach().float(),
                              self.model.head.bias.detach().float())
            loss = compute_training_loss(logits, targets, current, 1., 'legacy_full_zero')
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                             for p in self.parameters if p.grad is not None))
        metrics = self.model.interaction_bank[self.model.current_task_id].last_metrics
        self.values.append(dict(loss=float(loss.detach()), gradient_norm=norm, **metrics))

    def step(self):
        before = self.scaler.get_scale()
        self.scaler.step(self.optimizer); self.scaler.update()
        if self.scaler.get_scale() >= before:
            self.steps += 1
        else:
            self.skips += 1

    def end_epoch(self):
        if self.steps:
            self.scheduler.step()
        return {**{f'interaction_{k}':sum(v[k] for v in self.values)/len(self.values)
                    for k in self.values[0]}, 'interaction_steps':self.steps,
                'interaction_skipped_steps':self.skips}

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                    scaler=self.scaler.state_dict())

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
