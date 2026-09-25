"""Task-isolated feature-level fusion for Full/Person/Face views."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


class TaskwiseViewFusion(nn.Module):
    """Fuse matching MULTI-LANE task features without rewriting old lanes.

    A router belongs to exactly one incremental task.  During task ``t`` only
    router ``t`` is trainable; inference applies router ``k`` only to lane
    ``k``.  This mirrors selectors/prompts/Adapters and prevents a later task
    from changing fusion for previously introduced classes.
    """

    MODES = (
        "disabled",
        "fixed_three_view",
        "soft_three_view",
        "soft_full_person",
        "residual_full_person",
        "residual_three_view",
        "logit_residual_three_view",
    )
    THREE_VIEW_PRIOR = (0.64, 0.16, 0.20)
    INVALID_FACE_PRIOR = (0.80, 0.20, 0.0)
    TWO_VIEW_PRIOR = (0.80, 0.20)

    def __init__(
        self,
        num_tasks: int,
        feature_dim: int,
        mode: str,
        hidden_dim: int = 16,
        residual_scale: float = 0.1,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown taskwise view-fusion mode")
        if min(int(num_tasks), int(feature_dim), int(hidden_dim)) <= 0:
            raise ValueError("View-fusion dimensions must be positive")
        if not math.isfinite(residual_scale) or not 0 < residual_scale <= 1:
            raise ValueError("View residual scale must be finite and in (0, 1]")
        self.mode = mode
        self.num_tasks = int(num_tasks)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.num_classes = None if num_classes is None else int(num_classes)
        if self.logit_residual and (self.num_classes is None or self.num_classes <= 0):
            raise ValueError(
                "Logit residual calibration requires a positive class count"
            )
        self.current_task_id = -1
        self.view_names = (
            ("full", "person")
            if mode in {"soft_full_person", "residual_full_person"}
            else ("full", "person", "face")
        )
        self.task_routers = nn.ModuleList()
        self.task_residuals = nn.ModuleList()
        self.task_logit_residuals = nn.ParameterList()
        if mode.startswith("soft_"):
            view_count = len(self.view_names)
            input_dim = self.feature_dim * view_count
            prior = (
                self.TWO_VIEW_PRIOR
                if view_count == 2 else self.THREE_VIEW_PRIOR
            )
            with torch.random.fork_rng(devices=[]):
                for task_id in range(self.num_tasks):
                    # Deterministic task-local initialization without moving
                    # the global data/augmentation RNG stream.
                    torch.manual_seed(41_003 + task_id)
                    router = nn.Sequential(
                        nn.Linear(input_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, view_count),
                    )
                    nn.init.xavier_uniform_(router[0].weight)
                    nn.init.zeros_(router[0].bias)
                    nn.init.zeros_(router[2].weight)
                    with torch.no_grad():
                        router[2].bias.copy_(
                            torch.tensor([math.log(value) for value in prior])
                        )
                    self.task_routers.append(router)
        elif mode.startswith("residual_"):
            auxiliary_names = self.view_names[1:]
            with torch.random.fork_rng(devices=[]):
                for task_id in range(self.num_tasks):
                    torch.manual_seed(51_003 + task_id)
                    projections = nn.ModuleDict()
                    gates = nn.ParameterDict()
                    for name in auxiliary_names:
                        projection = nn.Sequential(
                            nn.LayerNorm(self.feature_dim, elementwise_affine=False),
                            nn.Linear(self.feature_dim, self.hidden_dim),
                            nn.GELU(),
                            nn.Linear(self.hidden_dim, self.feature_dim),
                        )
                        nn.init.xavier_uniform_(projection[1].weight)
                        nn.init.zeros_(projection[1].bias)
                        nn.init.zeros_(projection[3].weight)
                        nn.init.zeros_(projection[3].bias)
                        projections[name] = projection
                        gates[name] = nn.Parameter(torch.zeros(()))
                    self.task_residuals.append(nn.ModuleDict({
                        "projections": projections,
                        "gates": gates,
                    }))
        elif self.logit_residual:
            # One bounded Person coefficient and one bounded Face coefficient
            # per class.  Zero initialization makes the complete model exactly
            # equal to fixed three-view fusion before learning.
            for _ in range(self.num_tasks):
                self.task_logit_residuals.append(
                    nn.Parameter(torch.zeros(2, self.num_classes))
                )
        self.requires_grad_(False)

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @property
    def learned(self) -> bool:
        return self.mode.startswith(("soft_", "residual_")) or self.logit_residual

    @property
    def residual(self) -> bool:
        return self.mode.startswith("residual_")

    @property
    def logit_residual(self) -> bool:
        return self.mode == "logit_residual_three_view"

    def _task_modules(self):
        if self.logit_residual:
            return self.task_logit_residuals
        return self.task_residuals if self.residual else self.task_routers

    def restore_task(self, task_id: int) -> None:
        if not -1 <= int(task_id) < self.num_tasks:
            raise ValueError("Invalid view-fusion task")
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        if self.learned and task_id >= 0:
            modules = self._task_modules()
            modules[int(task_id)].requires_grad_(True)
        self.current_task_id = int(task_id)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self.learned and self.current_task_id >= 0:
            modules = self._task_modules()
            active = modules[self.current_task_id]
            if isinstance(active, nn.Parameter):
                yield active
            else:
                yield from active.parameters()

    def parameter_count_per_task(self) -> int:
        if not self.learned:
            return 0
        modules = self._task_modules()
        first = modules[0]
        if isinstance(first, nn.Parameter):
            return first.numel()
        return sum(parameter.numel() for parameter in first.parameters())

    def _validate(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None,
    ) -> Tuple[int, int, int]:
        if not self.enabled:
            raise RuntimeError("Disabled view fusion cannot fuse features")
        if set(features) != set(self.view_names):
            raise ValueError("View-fusion feature names do not match the mode")
        shapes = {tuple(value.shape) for value in features.values()}
        if len(shapes) != 1:
            raise ValueError("View-fusion feature shapes differ")
        shape = next(iter(shapes))
        if len(shape) != 3 or shape[1] != len(lane_ids) or shape[2] != self.feature_dim:
            raise ValueError("Invalid view-fusion lane feature shape")
        if len(set(int(value) for value in lane_ids)) != len(lane_ids):
            raise ValueError("View-fusion lane IDs must be unique")
        if any(not 0 <= int(value) < self.num_tasks for value in lane_ids):
            raise ValueError("View-fusion lane ID is outside the protocol")
        if len(self.view_names) == 3:
            if face_reliable is None or face_reliable.shape != (shape[0],):
                raise ValueError("Three-view fusion requires a per-sample Face mask")
        return shape

    def weights(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, lane_count, _ = self._validate(features, lane_ids, face_reliable)
        reference = features["full"]
        if self.residual:
            rows = []
            for task_id in lane_ids:
                module = self.task_residuals[int(task_id)]
                coefficients = [reference.new_ones(batch)]
                for name in self.view_names[1:]:
                    coefficient = self.residual_scale * torch.sigmoid(
                        module["gates"][name].float()
                    )
                    coefficient = coefficient.to(reference).expand(batch)
                    if name == "face":
                        coefficient = coefficient * face_reliable.to(
                            device=reference.device, dtype=reference.dtype
                        )
                    coefficients.append(coefficient)
                rows.append(torch.stack(coefficients, dim=-1))
            return torch.stack(rows, dim=1)
        if self.mode in {"fixed_three_view", "logit_residual_three_view"}:
            valid_prior = reference.new_tensor(self.THREE_VIEW_PRIOR)
            invalid_prior = reference.new_tensor(self.INVALID_FACE_PRIOR)
            return torch.where(
                face_reliable.to(device=reference.device, dtype=torch.bool)[:, None, None],
                valid_prior[None, None, :],
                invalid_prior[None, None, :],
            ).expand(batch, lane_count, -1)

        rows = []
        for lane_position, task_id in enumerate(lane_ids):
            router_parts = []
            for name in self.view_names:
                part = features[name][:, lane_position]
                if name == "face":
                    part = part * face_reliable.to(
                        device=part.device, dtype=part.dtype
                    )[:, None]
                router_parts.append(part)
            router_input = torch.cat(
                router_parts, dim=-1,
            )
            logits = self.task_routers[int(task_id)](router_input.float())
            if len(self.view_names) == 3:
                invalid = ~face_reliable.to(device=logits.device, dtype=torch.bool)
                logits = logits.clone()
                logits[invalid, 2] = torch.finfo(logits.dtype).min
            lane_weights = torch.softmax(logits, dim=-1)
            if not torch.isfinite(lane_weights).all():
                raise FloatingPointError("Non-finite taskwise fusion weights")
            rows.append(lane_weights)
        return torch.stack(rows, dim=1)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(features, lane_ids, face_reliable)
        if self.residual:
            rows = []
            for lane_position, task_id in enumerate(lane_ids):
                module = self.task_residuals[int(task_id)]
                fused = features["full"][:, lane_position]
                for view_position, name in enumerate(self.view_names[1:], start=1):
                    source = features[name][:, lane_position].detach().float()
                    raw_delta = module["projections"][name](source)
                    # Bound each auxiliary residual below unit L2 norm before
                    # applying the <= residual_scale coefficient.  Unlike a
                    # second normalization of the final feature, this keeps
                    # the zero-initialized output bitwise equal to Full.
                    delta = raw_delta / (
                        1.0 + torch.linalg.vector_norm(
                            raw_delta, dim=-1, keepdim=True
                        )
                    )
                    fused = fused + (
                        weights[:, lane_position, view_position, None]
                        .to(delta.dtype) * delta
                    )
                rows.append(fused)
            fused = torch.stack(rows, dim=1)
            if not torch.isfinite(fused).all():
                raise FloatingPointError("Non-finite taskwise residual fusion")
            return fused, weights
        stacked = torch.stack([features[name] for name in self.view_names], dim=2)
        fused = torch.sum(stacked * weights.unsqueeze(-1).to(stacked.dtype), dim=2)
        return fused, weights

    def calibrate_logits(
        self,
        base_logits: torch.Tensor,
        view_logits: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply a bounded task-local correction without changing B0 gradients.

        The base and per-view logits are detached deliberately.  Training uses
        a separate ordinary B0 loss for the shared representation/head and a
        calibration loss for these coefficients, so adding this module cannot
        redirect the B0 optimization path.
        """
        if not self.logit_residual:
            return base_logits
        if set(view_logits) != {"full", "person", "face"}:
            raise ValueError("Logit residual calibration requires three views")
        if base_logits.ndim != 3 or base_logits.shape[1] != len(lane_ids):
            raise ValueError("Invalid base logits for taskwise calibration")
        if any(value.shape != base_logits.shape for value in view_logits.values()):
            raise ValueError("Per-view logits do not match base logits")
        if face_reliable is None or face_reliable.shape != (base_logits.shape[0],):
            raise ValueError("Logit residual calibration requires a Face mask")

        base = base_logits.detach()
        person_delta = view_logits["person"].detach() - base
        face_delta = view_logits["face"].detach() - base
        face_mask = face_reliable.to(
            device=base.device, dtype=base.dtype
        )[:, None]
        rows = []
        for lane_position, task_id in enumerate(lane_ids):
            coefficients = self.residual_scale * torch.tanh(
                self.task_logit_residuals[int(task_id)].float()
            )
            person = coefficients[0].to(base) * person_delta[:, lane_position]
            face = (
                coefficients[1].to(base)
                * face_delta[:, lane_position]
                * face_mask
            )
            rows.append(base[:, lane_position] + person + face)
        calibrated = torch.stack(rows, dim=1)
        if not torch.isfinite(calibrated).all():
            raise FloatingPointError("Non-finite taskwise logit calibration")
        return calibrated

    def prior(self, face_reliable: bool = True) -> Dict[str, float]:
        if len(self.view_names) == 2:
            values = self.TWO_VIEW_PRIOR
        elif face_reliable:
            values = self.THREE_VIEW_PRIOR
        else:
            values = self.INVALID_FACE_PRIOR
        return dict(zip(self.view_names, values))
