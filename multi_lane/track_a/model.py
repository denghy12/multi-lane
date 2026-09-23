"""Independent Track-A MULTI-LANE model over frozen OpenAI CLIP blocks.

The implementation follows the fixed official release at ``5ee982c`` without
copying its source.  OpenAI CLIP replaces the released ImageNet ViT, while the
selector aggregation, task K/V prompts, drop-and-replace pathway, task-slice
copying, shared classifier, and concat inference remain method-specific.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .adapter import (
    ParaXImageAdapterBank, TaskImageTokenAdapterBank, TaskLaneTransformerAdapterBank,
)
from .selector_conditioning import TaskSelectorConditioner
from .view_fusion import TaskwiseViewFusion

ModelInputs = Union[torch.Tensor, Dict[str, torch.Tensor]]


class MultiLaneModel(nn.Module):
    """Frozen CLIP visual tower with preallocated MULTI-LANE task pathways."""

    def __init__(
        self,
        visual_encoder: nn.Module,
        task_sizes: Sequence[int],
        num_selectors: int = 10,
        num_prompts: int = 10,
        num_prompt_layers: int = 5,
        selector_mode: str = "shared",
        selector_view_residual_scale: float = 0.1,
        prompt_mode: str = "shared",
        prompt_private_layers: int = 0,
        normalize: str = "pre-head",
        adapter_mode: str = "disabled",
        adapter_bottleneck_dim: int = 64,
        adapter_layer_indices: Sequence[int] = (11,),
        adapter_residual_scale: float = 0.1,
        adapter_activation: str = "relu",
        adapter_task_initialization: str = "independent",
        adapter_bottleneck_dims_per_task: Optional[Sequence[int]] = None,
        adapter_residual_gate_mode: str = "fixed",
        adapter_auxiliary_metric_mode: str = "none",
        adapter_view_bottleneck_dim: int = 0,
        adapter_view_mode: str = "shared",
        selector_conditioning: str = "disabled",
        selector_condition_layers: Sequence[int] = (1,),
        selector_condition_hidden_dim: int = 32,
        selector_condition_scale: float = 0.1,
        view_fusion: str = "disabled",
        view_fusion_hidden_dim: int = 16,
        view_residual_scale: float = 0.1,
        detach_view_fusion_features: bool = False,
        view_classifier_mode: str = "shared_post_fusion",
        parax_mode: str = "disabled",
        parax_rank: int = 32,
        parax_num_experts: int = 3,
        parax_layer_indices: Sequence[int] = (10,),
        parax_router_hidden: int = 16,
        parax_residual_scale: float = 0.1,
        parax_level_conditioned: bool = False,
        parax_initialization: str = "official",
    ) -> None:
        super().__init__()
        if not task_sizes or any(int(size) <= 0 for size in task_sizes):
            raise ValueError("MULTI-LANE task sizes must be positive")
        if num_selectors <= 0 or num_prompts <= 0:
            raise ValueError("MULTI-LANE selector/prompt counts must be positive")
        if selector_mode not in {"shared", "view_specific", "shared_residual"}:
            raise ValueError(
                "MULTI-LANE selector mode must be shared, view_specific, "
                "or shared_residual"
            )
        if not 0 < float(selector_view_residual_scale) <= 1:
            raise ValueError("Selector view residual scale must be in (0, 1]")
        if prompt_mode not in {
            "shared", "view_specific", "view_specific_full_face",
            "late_view_residual_full_face",
        }:
            raise ValueError(
                "MULTI-LANE prompt mode must be shared, view_specific, "
                "view_specific_full_face, or late_view_residual_full_face"
            )
        if not 0 <= int(prompt_private_layers) <= int(num_prompt_layers):
            raise ValueError("Prompt private layer count is invalid")
        if normalize not in {"none", "pre-head"}:
            raise ValueError("MULTI-LANE normalize must be none or pre-head")
        if adapter_mode not in {"disabled", "task_lane", "image_token"}:
            raise ValueError(
                "Adapter mode must be disabled, task_lane, or image_token"
            )
        if parax_mode not in {"disabled", "image", "image_level", "static", "post"}:
            raise ValueError("Invalid ParaX mode")
        if int(adapter_view_bottleneck_dim) < 0:
            raise ValueError("View-specific Adapter bottleneck must be non-negative")
        if adapter_view_mode not in {"shared", "independent"}:
            raise ValueError("Image-token Adapter view mode must be shared or independent")
        if adapter_view_mode == "independent" and int(adapter_view_bottleneck_dim) <= 0:
            raise ValueError("Independent Adapter views require a positive view bottleneck")
        if adapter_view_bottleneck_dim and adapter_mode != "image_token":
            raise ValueError(
                "View-specific specialization requires Image-token Adapter mode"
            )
        required = (
            "conv1",
            "class_embedding",
            "positional_embedding",
            "ln_pre",
            "transformer",
            "ln_post",
            "proj",
        )
        missing = [name for name in required if not hasattr(visual_encoder, name)]
        if missing:
            raise TypeError(
                "MULTI-LANE Track A requires an OpenAI CLIP VisionTransformer; "
                "missing " + ", ".join(missing)
            )
        if not hasattr(visual_encoder.transformer, "resblocks"):
            raise TypeError("CLIP visual transformer must expose residual blocks")
        blocks = list(visual_encoder.transformer.resblocks)
        if not blocks:
            raise ValueError("CLIP visual transformer has no residual blocks")
        if selector_conditioning not in TaskSelectorConditioner.MODES:
            raise ValueError("Invalid selector conditioning mode")
        if view_fusion not in TaskwiseViewFusion.MODES:
            raise ValueError("Invalid taskwise view-fusion mode")
        if view_classifier_mode not in {
            "shared_post_fusion", "shared_per_view", "full_private_per_view",
            "private_per_view",
        }:
            raise ValueError("Invalid view classifier mode")
        if (
            view_classifier_mode != "shared_post_fusion"
            and view_fusion != "fixed_three_view"
        ):
            raise ValueError(
                "Per-view classification requires fixed three-view fusion"
            )
        if view_fusion != "disabled" and selector_conditioning != "disabled":
            raise ValueError("View fusion and Selector conditioning are mutually exclusive")
        if selector_conditioning != "disabled" and (
            not selector_condition_layers
            or any(i < 0 or i >= len(blocks) for i in selector_condition_layers)
        ):
            raise ValueError("Selector conditioning layer is outside the visual transformer")
        if not 0 <= num_prompt_layers <= len(blocks):
            raise ValueError("MULTI-LANE prompt-layer count is invalid")
        adapter_layers = tuple(int(index) for index in adapter_layer_indices)
        if adapter_mode != "disabled" and (
            not adapter_layers
            or any(index < 0 or index >= len(blocks) for index in adapter_layers)
        ):
            raise ValueError("Adapter layer index is outside the visual transformer")

        width = int(visual_encoder.conv1.out_channels)
        output_dim = int(getattr(visual_encoder, "output_dim", -1))
        if width <= 0 or output_dim <= 0:
            raise ValueError("CLIP visual dimensions must be positive")
        first_attention = blocks[0].attn
        num_heads = int(first_attention.num_heads)
        if width % num_heads:
            raise ValueError("CLIP width must be divisible by attention heads")
        for block in blocks:
            block_width = int(block.attn.embed_dim)
            if block_width != width or int(block.attn.num_heads) != num_heads:
                raise ValueError("CLIP visual residual blocks are inconsistent")

        self.visual_encoder = visual_encoder
        self.visual_encoder.requires_grad_(False)
        self.width = width
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.num_selectors = int(num_selectors)
        self.selector_mode = selector_mode
        self.selector_view_names = ("full", "person", "face")
        self.selector_view_residual_scale = float(selector_view_residual_scale)
        self.prompt_mode = prompt_mode
        self.prompt_private_layers = int(prompt_private_layers)
        self.num_prompts = int(num_prompts)
        self.num_prompt_layers = int(num_prompt_layers)
        self.normalize = normalize
        self.adapter_mode = adapter_mode
        self.adapter_runtime_enabled = adapter_mode != "disabled"
        self.parax_mode = parax_mode
        self.parax_runtime_enabled = parax_mode != "disabled"
        self.parax_bank = None
        self._parax_gate_records = []
        if self.parax_runtime_enabled:
            self.parax_bank = ParaXImageAdapterBank(
                hidden_dim=self.width, rank=parax_rank,
                num_experts=parax_num_experts, layer_indices=parax_layer_indices,
                router_hidden=parax_router_hidden, residual_scale=parax_residual_scale,
                level_conditioned=(parax_level_conditioned or parax_mode == "image_level"),
                static=(parax_mode == "static"),
                initialization=parax_initialization,
            )
        self._task_sizes = tuple(int(size) for size in task_sizes)
        self._current_task_id = -1

        # Draw one shared bank first in both modes.  In view-specific mode the
        # three banks are exact copies at initialization, so the new pathway is
        # numerically equivalent to the historical shared Selector before any
        # optimization and does not consume extra global RNG state.
        selectors = torch.randn(
            len(self._task_sizes), self.num_selectors, self.width
        )
        nn.init.orthogonal_(selectors)
        if selector_mode == "view_specific":
            selectors = selectors.unsqueeze(1).expand(
                -1, len(self.selector_view_names), -1, -1
            ).clone()
        self.selectors = nn.Parameter(selectors)
        self.selector_view_residuals = (
            nn.Parameter(torch.zeros(
                len(self._task_sizes), len(self.selector_view_names),
                self.num_selectors, self.width,
            ))
            if selector_mode == "shared_residual" else None
        )

        prompts = nn.ParameterList()
        for layer_index in range(self.num_prompt_layers):
            value = torch.randn(
                2,
                len(self._task_sizes),
                self.num_prompts,
                self.num_heads,
                self.head_dim,
            )
            nn.init.orthogonal_(value)
            late_private = (
                prompt_mode == "late_view_residual_full_face"
                and layer_index >= int(num_prompt_layers) - int(prompt_private_layers)
            )
            if prompt_mode in {"view_specific", "view_specific_full_face"} or late_private:
                value = value.unsqueeze(1).expand(
                    -1, len(self.selector_view_names), -1, -1, -1, -1
                ).clone()
            prompts.append(nn.Parameter(value))
        self.prompts = prompts

        self.head = nn.Linear(self.output_dim, self.num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.view_classifier_mode = view_classifier_mode
        self.private_view_heads = nn.ModuleList()
        if view_classifier_mode == "private_per_view":
            # Keep the Full head in ``self.head`` for backward-compatible
            # state names; Person and Face receive exact initialization copies.
            for _ in self.selector_view_names[1:]:
                self.private_view_heads.append(copy.deepcopy(self.head))
        self.full_view_head = (
            copy.deepcopy(self.head)
            if view_classifier_mode == "full_private_per_view" else None
        )

        mask = torch.zeros(len(self._task_sizes), self.num_classes)
        offset = 0
        for task_id, size in enumerate(self._task_sizes):
            mask[task_id, offset : offset + size] = 1.0
            offset += size
        self.register_buffer("task_class_mask", mask, persistent=True)

        self.adapter_bank = None
        if self.adapter_mode != "disabled":
            # Adapter initialization must not perturb the global RNG stream
            # that drives DataLoader shuffling and stochastic transforms.  The
            # bank still receives deterministic seed-specific initialization,
            # while code after model construction observes the same RNG state
            # as the adapter-disabled baseline.
            with torch.random.fork_rng(devices=[]):
                bank_class = (
                    TaskImageTokenAdapterBank
                    if self.adapter_mode == "image_token"
                    else TaskLaneTransformerAdapterBank
                )
                adapter_bank = bank_class(
                    num_tasks=len(self._task_sizes),
                    hidden_dim=self.width,
                    bottleneck_dim=adapter_bottleneck_dim,
                    layer_indices=adapter_layers,
                    residual_scale=adapter_residual_scale,
                    activation=adapter_activation,
                    task_initialization=adapter_task_initialization,
                    bottleneck_dims_per_task=adapter_bottleneck_dims_per_task,
                    residual_gate_mode=adapter_residual_gate_mode,
                    auxiliary_metric_mode=adapter_auxiliary_metric_mode,
                    **(
                        {"view_bottleneck_dim": adapter_view_bottleneck_dim}
                        if self.adapter_mode == "image_token" else {}
                    ),
                    **(
                        {"view_mode": adapter_view_mode}
                        if self.adapter_mode == "image_token" else {}
                    ),
                )
            self.adapter_bank = adapter_bank

        self.selector_conditioning = selector_conditioning
        self.selector_conditioning_runtime_enabled = selector_conditioning != "disabled"
        self.selector_conditioner = None
        if selector_conditioning != "disabled":
            with torch.random.fork_rng(devices=[]):
                self.selector_conditioner = TaskSelectorConditioner(
                    len(self._task_sizes), self.width, selector_conditioning,
                    selector_condition_layers, selector_condition_hidden_dim,
                    selector_condition_scale,
                )

        self.view_fusion = view_fusion
        self.detach_view_fusion_features = bool(detach_view_fusion_features)
        if self.detach_view_fusion_features and view_fusion == "disabled":
            raise ValueError(
                "View-fusion feature detachment requires enabled view fusion"
            )
        with torch.random.fork_rng(devices=[]):
            self.view_fusion_module = TaskwiseViewFusion(
                len(self._task_sizes), self.output_dim, view_fusion,
                hidden_dim=view_fusion_hidden_dim,
                residual_scale=view_residual_scale,
            )
        self._last_fusion_weights: Optional[torch.Tensor] = None

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        return self._task_sizes

    @property
    def num_tasks(self) -> int:
        return len(self._task_sizes)

    @property
    def num_classes(self) -> int:
        return sum(self._task_sizes)

    @property
    def current_task_id(self) -> int:
        return self._current_task_id

    @property
    def seen_classes(self) -> int:
        if self._current_task_id < 0:
            return 0
        return sum(self._task_sizes[: self._current_task_id + 1])

    def activate_task(self, task_id: int) -> None:
        expected = self._current_task_id + 1
        if task_id != expected:
            raise RuntimeError(
                f"MULTI-LANE tasks must be sequential: expected {expected}, "
                f"got {task_id}"
            )
        if not 0 <= task_id < self.num_tasks:
            raise ValueError("MULTI-LANE task id is outside the protocol")
        if task_id > 0:
            with torch.no_grad():
                self.selectors[task_id].copy_(self.selectors[task_id - 1])
                if self.selector_view_residuals is not None:
                    self.selector_view_residuals[task_id].copy_(
                        self.selector_view_residuals[task_id - 1]
                    )
                for prompt in self.prompts:
                    if prompt.ndim == 6:
                        prompt[:, :, task_id].copy_(prompt[:, :, task_id - 1])
                    else:
                        prompt[:, task_id].copy_(prompt[:, task_id - 1])
        if self.adapter_bank is not None:
            self.adapter_bank.activate_task(task_id)
        if self.parax_bank is not None:
            self.parax_bank.activate_task(task_id)
        if self.selector_conditioner is not None:
            self.selector_conditioner.restore_task(task_id)
        self.view_fusion_module.restore_task(task_id)
        self._current_task_id = int(task_id)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < self.num_tasks:
            raise ValueError("MULTI-LANE restored task id is invalid")
        if self.adapter_bank is not None:
            self.adapter_bank.restore_task(task_id)
        if self.parax_bank is not None:
            self.parax_bank.restore_task(task_id)
        if self.selector_conditioner is not None:
            self.selector_conditioner.restore_task(task_id)
        self.view_fusion_module.restore_task(task_id)
        self._current_task_id = int(task_id)

    def set_adapter_runtime_enabled(self, enabled: bool) -> None:
        if enabled and self.adapter_bank is None:
            raise RuntimeError("Cannot enable an adapter that was not configured")
        self.adapter_runtime_enabled = bool(enabled)

    def set_parax_runtime_enabled(self, enabled: bool) -> None:
        if enabled and self.parax_bank is None:
            raise RuntimeError("Cannot enable ParaX that was not configured")
        self.parax_runtime_enabled = bool(enabled)

    def _record_parax_gates(
        self, layer_id: int, image_view: str, before: torch.Tensor,
        after: torch.Tensor, gates: torch.Tensor,
    ) -> None:
        if not self.parax_runtime_enabled:
            return
        token_norm = before.detach().float().norm(dim=-1).mean().clamp_min(1e-12)
        residual_norm = (after.detach().float() - before.detach().float()).norm(dim=-1).mean()
        self._parax_gate_records.append({
            "layer_id": int(layer_id),
            "view": str(image_view),
            "gates": gates.detach().float(),
            "residual_ratio": float((residual_norm / token_norm).cpu()),
        })

    def parax_gate_diagnostics(self) -> Dict[str, float]:
        """Return per-view/layer gate and residual diagnostics for the last forward."""
        if not self._parax_gate_records:
            return {}
        result: Dict[str, float] = {}
        grouped: Dict[Tuple[str, int], list[dict]] = {}
        for record in self._parax_gate_records:
            grouped.setdefault((record["view"], record["layer_id"]), []).append(record)
        for (view, layer_id), records in grouped.items():
            gates = torch.cat([record["gates"] for record in records], dim=0)
            prefix = f"parax_{view}_layer{layer_id}"
            mean = gates.mean(dim=0)
            entropy = -(gates.clamp_min(1e-12) * gates.clamp_min(1e-12).log()).sum(dim=-1).mean()
            top_frequency = torch.bincount(gates.argmax(dim=-1), minlength=gates.shape[1]).float() / gates.shape[0]
            for expert, value in enumerate(mean):
                result[f"{prefix}_gate_mean_e{expert}"] = float(value)
            for expert, value in enumerate(top_frequency):
                result[f"{prefix}_top_frequency_e{expert}"] = float(value)
            result[f"{prefix}_entropy"] = float(entropy)
            result[f"{prefix}_residual_ratio"] = float(
                sum(record["residual_ratio"] for record in records) / len(records)
            )
        views = sorted({record["view"] for record in self._parax_gate_records})
        if len(views) >= 2:
            view_means = {}
            for view in views:
                view_records = [r for r in self._parax_gate_records if r["view"] == view]
                view_means[view] = torch.cat([r["gates"] for r in view_records], dim=0).mean(dim=0)
            distances = []
            for index, left in enumerate(views):
                for right in views[index + 1:]:
                    distances.append(torch.norm(view_means[left] - view_means[right], p=1))
            result["parax_view_gate_l1_distance"] = float(torch.stack(distances).mean())
        return result

    def set_selector_conditioning_runtime_enabled(self, enabled: bool) -> None:
        if enabled and self.selector_conditioner is None:
            raise RuntimeError("Cannot enable unconfigured selector conditioning")
        self.selector_conditioning_runtime_enabled = bool(enabled)

    def _visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder
        x = images.to(dtype=visual.conv1.weight.dtype)
        with torch.no_grad():
            x = visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1)
            x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
            positional = visual.positional_embedding.to(x.dtype)
            if positional.shape[0] != x.shape[1]:
                raise ValueError(
                    "CLIP positional-token count differs from the input patches"
                )
            x = visual.ln_pre(x + positional)
        return x

    def _lane_ids(self, all_seen_lanes: bool) -> List[int]:
        if self._current_task_id < 0:
            raise RuntimeError("No MULTI-LANE task is active")
        if all_seen_lanes:
            return list(range(self._current_task_id + 1))
        return [self._current_task_id]

    def _initial_lane_tokens(
        self, batch_size: int, lane_ids: Sequence[int], image_view: str = "full"
    ) -> torch.Tensor:
        if image_view not in self.selector_view_names:
            raise ValueError(f"Unknown Selector view: {image_view}")
        selector = self.selectors[list(lane_ids)]
        if self.selector_mode == "view_specific":
            selector = selector[:, self.selector_view_names.index(image_view)]
        elif self.selector_mode == "shared_residual":
            selector = selector + self.selector_view_residual_scale * (
                self.selector_view_residuals[
                    list(lane_ids), self.selector_view_names.index(image_view)
                ]
            )
        selector = selector.unsqueeze(1).expand(-1, batch_size, -1, -1)
        cls = self.visual_encoder.class_embedding.to(selector.dtype)
        cls = cls.reshape(1, 1, 1, -1).expand(
            len(lane_ids), batch_size, 1, -1
        )
        return torch.cat([cls, selector], dim=2)

    def _prompt_attention(
        self,
        block: nn.Module,
        task_tokens: torch.Tensor,
        lane_ids: Sequence[int],
        layer_id: int,
        image_view: str = "full",
    ) -> torch.Tensor:
        task_count, batch, token_count, width = task_tokens.shape
        attention = block.attn
        qkv = F.linear(
            task_tokens,
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        qkv = qkv.reshape(
            task_count,
            batch,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(3, 0, 1, 4, 2, 5)
        query, key, value = qkv.unbind(0)
        if layer_id < self.num_prompt_layers:
            prompt = self.prompts[layer_id]
            if prompt.ndim == 6:
                if image_view not in self.selector_view_names:
                    raise ValueError(f"Unknown Prompt view: {image_view}")
                # Selective mode reserves bank 0 as the historical shared
                # Person bank; Full and Face use private banks 1 and 2.
                if self.prompt_mode == "view_specific":
                    view_index = self.selector_view_names.index(image_view)
                elif self.prompt_mode in {
                    "view_specific_full_face", "late_view_residual_full_face"
                }:
                    view_index = {"person": 0, "full": 1, "face": 2}[image_view]
                prompt = prompt[:, view_index, list(lane_ids)]
            else:
                prompt = prompt[:, list(lane_ids)]
            prompt = prompt.permute(0, 1, 3, 2, 4)
            prompt = prompt.unsqueeze(2).expand(-1, -1, batch, -1, -1, -1)
            key = torch.cat([prompt[0], key], dim=-2)
            value = torch.cat([prompt[1], value], dim=-2)
        weights = torch.matmul(query, key.transpose(-2, -1))
        weights = torch.softmax(weights * (self.head_dim**-0.5), dim=-1)
        output = torch.matmul(weights, value)
        output = output.permute(0, 1, 3, 2, 4).reshape(
            task_count, batch, token_count, width
        )
        return F.linear(
            output,
            attention.out_proj.weight,
            attention.out_proj.bias,
        )

    def _lane_block(
        self,
        block: nn.Module,
        image_tokens: torch.Tensor,
        lane_tokens: torch.Tensor,
        lane_ids: Sequence[int],
        layer_id: int,
        query_delta: Optional[torch.Tensor] = None,
        person_tokens: Optional[torch.Tensor] = None,
        person_patch_mask: Optional[torch.Tensor] = None,
        condition_valid: Optional[torch.Tensor] = None,
        image_view: str = "full",
    ) -> torch.Tensor:
        # The released block applies its first LayerNorm before both selector
        # aggregation and prompt attention.  Keep the residual stream itself
        # unnormalized, as in the original pre-norm transformer.
        frozen_normalized_image = block.ln_1(image_tokens)
        if not self.parax_runtime_enabled:
            frozen_normalized_image = frozen_normalized_image.detach()
        selector_image_tokens = frozen_normalized_image
        normalized_lane = block.ln_1(lane_tokens)
        task_cls = normalized_lane[:, :, :1]
        selectors = normalized_lane[:, :, 1:]
        if (self.selector_conditioning == "person_patches"
                and self.selector_conditioning_runtime_enabled
                and layer_id in self.selector_conditioner.layer_indices):
            if (person_tokens is None or person_patch_mask is None
                    or condition_valid is None):
                raise ValueError("Person-patch conditioning inputs are missing")
            normalized_person_patches = block.ln_1(person_tokens)[:, 1:].detach()
            query_delta = self.selector_conditioner.patch_query_delta(
                layer_id, lane_ids, selectors, normalized_person_patches,
                person_patch_mask, condition_valid,
            )
        # Condition only the query used to read image tokens. Drop-and-replace
        # must still restore the ORIGINAL selectors, not the modified queries.
        queries = selectors if query_delta is None else selectors + query_delta.to(selectors.dtype)
        if self.adapter_mode == "image_token" and self.adapter_runtime_enabled:
            selector_image_tokens = self.adapter_bank.adapted_tokens_for_layer(
                layer_id, frozen_normalized_image, lane_ids, image_view
            )
            similarity = torch.einsum(
                "tbsc,tbnc->tbsn", queries, selector_image_tokens
            ) * (self.width**-0.5)
            selected = torch.einsum(
                "tbnc,tbsn->tbsc",
                selector_image_tokens,
                torch.softmax(similarity, dim=-1),
            )
        else:
            # Preserve the historical contraction path exactly for disabled
            # and task-lane modes.
            similarity = torch.einsum(
                "tbsc,bnc->tbsn", queries, selector_image_tokens
            ) * (self.width**-0.5)
            selected = torch.einsum(
                "bnc,tbsn->tbsc",
                selector_image_tokens,
                torch.softmax(similarity, dim=-1),
            )
        summarized = torch.cat([task_cls, selected], dim=2)
        attention_output = self._prompt_attention(
            block,
            summarized,
            lane_ids,
            layer_id,
            image_view=image_view,
        )
        # Released drop-and-replace: retain the attended CLS update and put the
        # selector tokens back before the residual addition.
        update = torch.cat([attention_output[:, :, :1], selectors], dim=2)
        lane_tokens = lane_tokens + update
        normalized_residual = block.ln_2(lane_tokens)
        lane_tokens = lane_tokens + block.mlp(normalized_residual)
        if self.adapter_mode == "task_lane" and self.adapter_runtime_enabled:
            lane_tokens = lane_tokens + self.adapter_bank.delta_for_layer(
                layer_id, normalized_residual, lane_ids
            )
        return lane_tokens

    def _encode_single_lanes(
        self, images: ModelInputs, all_seen_lanes: bool, image_view: str = "full"
    ) -> torch.Tensor:
        lane_ids = self._lane_ids(all_seen_lanes)
        person_descriptor = None
        person_tokens = None
        condition_enabled = (self.selector_conditioner is not None
                             and self.selector_conditioning_runtime_enabled)
        paired = images if isinstance(images, dict) else None
        if paired is not None:
            images = paired["full"]
        if condition_enabled:
            required = {"person", "bbox", "condition_valid"}
            if self.selector_conditioning == "person_patches":
                required.add("person_patch_mask")
            if paired is None or not required <= paired.keys():
                raise ValueError("Enabled selector conditioning requires paired Full/Person inputs")
            batch_size = images.shape[0]
            if (paired["person"].shape != images.shape
                    or paired["bbox"].shape != (batch_size, 6)
                    or paired["condition_valid"].shape != (batch_size,)):
                raise ValueError("Invalid paired Full/Person batch shapes")
            if self.selector_conditioning == "person_patches":
                person_tokens = self._visual_tokens(paired["person"])
                if paired["person_patch_mask"].shape != person_tokens[:, 1:].shape[:2]:
                    raise ValueError("Person patch mask does not match CLIP patch tokens")
            elif "person" in self.selector_conditioning:
                # Full frozen CLIP CLS (post-ln, pre-projection), without task
                # adapters or a Person classifier. No future-task supervision.
                with torch.no_grad():
                    person_tokens = self._visual_tokens(paired["person"])
                    for block in self.visual_encoder.transformer.resblocks:
                        person_tokens = block(person_tokens.permute(1, 0, 2)).permute(1, 0, 2)
                    person_descriptor = self.visual_encoder.ln_post(person_tokens[:, 0]).float()
        if self.adapter_bank is not None:
            self.adapter_bank.reset_auxiliary_metrics()
        image_tokens = self._visual_tokens(images)
        lane_tokens = self._initial_lane_tokens(
            images.shape[0], lane_ids, image_view=image_view
        )
        for layer_id, block in enumerate(self.visual_encoder.transformer.resblocks):
            post_mode_last_layer = (
                self.parax_runtime_enabled and self.parax_bank is not None
                and self.parax_mode == "post"
                and layer_id == len(self.visual_encoder.transformer.resblocks) - 1
            )
            if post_mode_last_layer:
                with torch.no_grad():
                    image_tokens = block(image_tokens.permute(1, 0, 2)).permute(1, 0, 2)
            if (self.parax_runtime_enabled and self.parax_bank is not None
                    and self.parax_mode != "post"
                    and layer_id > 0
                    and layer_id - 1 in self.parax_bank.layer_indices):
                patch_tokens = image_tokens[:, 1:]
                before_patch_tokens = patch_tokens
                patch_tokens, gates = self.parax_bank(
                    layer_id - 1, patch_tokens, view_name=image_view
                )
                image_tokens = torch.cat([image_tokens[:, :1], patch_tokens], dim=1)
                self._last_parax_gates = gates.detach()
                self._record_parax_gates(
                    layer_id - 1, image_view, before_patch_tokens, patch_tokens, gates
                )
            if (self.parax_runtime_enabled and self.parax_bank is not None
                    and post_mode_last_layer):
                patch_tokens = image_tokens[:, 1:]
                before_patch_tokens = patch_tokens
                patch_tokens, gates = self.parax_bank(
                    self.parax_bank.layer_indices[0], patch_tokens, view_name=image_view
                )
                image_tokens = torch.cat([image_tokens[:, :1], patch_tokens], dim=1)
                self._last_parax_gates = gates.detach()
                self._record_parax_gates(
                    self.parax_bank.layer_indices[0], image_view, before_patch_tokens,
                    patch_tokens, gates
                )
            query_delta = None
            if (condition_enabled
                    and self.selector_conditioning != "person_patches"
                    and layer_id in self.selector_conditioner.layer_indices):
                query_delta = self.selector_conditioner.query_delta(
                    layer_id, lane_ids, person_descriptor,
                    paired["bbox"], paired["condition_valid"],
                )
            lane_tokens = self._lane_block(
                block,
                image_tokens,
                lane_tokens,
                lane_ids,
                layer_id,
                query_delta,
                person_tokens,
                paired["person_patch_mask"] if person_tokens is not None else None,
                paired["condition_valid"] if person_tokens is not None else None,
                image_view,
            )
            if post_mode_last_layer:
                pass  # The frozen final CLIP block already ran before post-encoder ParaX.
            elif not self.parax_runtime_enabled:
                with torch.no_grad():
                    image_tokens = block(image_tokens.permute(1, 0, 2)).permute(
                        1, 0, 2
                    )
            else:
                image_tokens = block(image_tokens.permute(1, 0, 2)).permute(1, 0, 2)
                if (person_tokens is not None
                        and layer_id < max(self.selector_conditioner.layer_indices)):
                    person_tokens = block(
                        person_tokens.permute(1, 0, 2)
                    ).permute(1, 0, 2)
        lane_tokens = self.visual_encoder.ln_post(lane_tokens)
        if self.visual_encoder.proj is not None:
            lane_tokens = lane_tokens @ self.visual_encoder.proj
        lane_cls = lane_tokens[:, :, 0].permute(1, 0, 2).float()
        if self.normalize == "pre-head":
            lane_cls = F.normalize(lane_cls, dim=-1)
        return lane_cls

    def encode_lanes_with_views(
        self, images: ModelInputs, all_seen_lanes: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return fused lane features and their supervised view features."""
        self._parax_gate_records = []
        if self.view_fusion == "disabled":
            self._last_fusion_weights = None
            return self._encode_single_lanes(images, all_seen_lanes), {}
        if not isinstance(images, dict):
            raise ValueError("Enabled view fusion requires dictionary inputs")
        required = set(self.view_fusion_module.view_names)
        if not required <= images.keys():
            raise ValueError("Enabled view fusion is missing image views")
        batch = images["full"].shape[0]
        if any(images[name].shape != images["full"].shape for name in required):
            raise ValueError("View-fusion images have inconsistent shapes")
        face_reliable = None
        if "face" in required:
            face_reliable = images.get("face_reliable")
            if face_reliable is None or face_reliable.shape != (batch,):
                raise ValueError("Three-view fusion requires a valid Face mask")
        if self.view_fusion_module.residual:
            # The auxiliary views are complementary frozen feature sources.
            # Their task-specific residual modules learn the mapping, while
            # only Full is allowed to update the shared MULTI-LANE pathway.
            adapter_runtime = self.adapter_runtime_enabled
            try:
                # Do not run the shared Full Image-token Adapter in auxiliary
                # branches. Besides enforcing view specialization, this avoids
                # AMP's cast cache retaining no-grad Adapter weights before the
                # differentiable Full pass in the same autocast context.
                if self.adapter_bank is not None:
                    self.set_adapter_runtime_enabled(False)
                with torch.no_grad():
                    features = {
                        name: self._encode_single_lanes(
                            images[name], all_seen_lanes, image_view=name
                        )
                        for name in self.view_fusion_module.view_names[1:]
                    }
            finally:
                self.set_adapter_runtime_enabled(adapter_runtime)
            features["full"] = self._encode_single_lanes(
                images["full"], all_seen_lanes, image_view="full"
            )
        else:
            features = {
                name: self._encode_single_lanes(
                    images[name], all_seen_lanes, image_view=name
                )
                for name in self.view_fusion_module.view_names
            }
        lane_ids = self._lane_ids(all_seen_lanes)
        fusion_features = (
            {name: value.detach() for name, value in features.items()}
            if self.detach_view_fusion_features
            and not self.view_fusion_module.residual
            else features
        )
        fused, weights = self.view_fusion_module(
            fusion_features, lane_ids, face_reliable
        )
        self._last_fusion_weights = weights.detach()
        return fused, features

    def encode_lanes(
        self, images: ModelInputs, all_seen_lanes: bool
    ) -> torch.Tensor:
        fused, _ = self.encode_lanes_with_views(images, all_seen_lanes)
        return fused

    def current_all_logits_with_views(
        self, images: ModelInputs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, view_logits, _ = self.current_all_logits_with_view_features(images)
        return logits, view_logits

    def current_all_logits_with_view_features(
        self, images: ModelInputs
    ) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """Return current logits and branch endpoints for gradient diagnostics."""
        fused, features = self.encode_lanes_with_views(images, all_seen_lanes=False)
        view_lane_logits = {
            name: self._head_for_view(name)(value)
            for name, value in features.items()
        }
        if self.view_classifier_mode == "shared_post_fusion":
            fused_lane_logits = self.head(fused)
        else:
            fused_lane_logits = self._fuse_view_lane_logits(view_lane_logits)
        return (
            fused_lane_logits[:, 0],
            {name: value[:, 0] for name, value in view_lane_logits.items()},
            features,
        )

    def _head_for_view(self, name: str) -> nn.Linear:
        if name not in self.view_fusion_module.view_names:
            raise ValueError(f"Unknown fused view: {name}")
        if self.view_classifier_mode == "private_per_view":
            index = self.selector_view_names.index(name)
            return self.head if index == 0 else self.private_view_heads[index - 1]
        if name == "full" and self.full_view_head is not None:
            return self.full_view_head
        return self.head

    def _fuse_view_lane_logits(
        self, view_lane_logits: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        weights = self._last_fusion_weights
        if weights is None:
            raise RuntimeError("Per-view logit fusion requires current weights")
        names = self.view_fusion_module.view_names
        if set(view_lane_logits) != set(names):
            raise ValueError("Per-view logits do not match configured views")
        stacked = torch.stack([view_lane_logits[name] for name in names], dim=2)
        return torch.sum(
            stacked * weights.unsqueeze(-1).to(dtype=stacked.dtype), dim=2
        )

    def seen_logits_with_views(
        self, images: ModelInputs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return fused and per-view logits with task-lane class ownership."""
        lane_ids = self._lane_ids(all_seen_lanes=True)
        fused, features = self.encode_lanes_with_views(
            images, all_seen_lanes=True
        )
        masks = self.task_class_mask[lane_ids].to(dtype=fused.dtype)

        def combine(logits: torch.Tensor) -> torch.Tensor:
            combined = torch.sum(logits * masks.unsqueeze(0), dim=1)
            return combined[:, : self.seen_classes]
        view_lane_logits = {
            name: self._head_for_view(name)(value)
            for name, value in features.items()
        }
        fused_lane_logits = (
            self.head(fused)
            if self.view_classifier_mode == "shared_post_fusion"
            else self._fuse_view_lane_logits(view_lane_logits)
        )
        return combine(fused_lane_logits), {
            name: combine(value) for name, value in view_lane_logits.items()
        }

    def last_fusion_weights(self) -> Optional[torch.Tensor]:
        if self._last_fusion_weights is None:
            return None
        return self._last_fusion_weights.detach()

    def fusion_weight_means(self) -> Optional[Tuple[float, ...]]:
        if self._last_fusion_weights is None:
            return None
        values = self._last_fusion_weights.float().mean(dim=(0, 1)).cpu()
        return tuple(float(value) for value in values)

    def adapter_auxiliary_metric(self, mode: str) -> torch.Tensor:
        if self.adapter_mode != "image_token" or self.adapter_bank is None:
            raise RuntimeError(
                "Adapter output regularization requires an Image-token Adapter"
            )
        return self.adapter_bank.auxiliary_metric(mode)

    def selector_view_residual_metric(self) -> torch.Tensor:
        """Mean squared magnitude of optional view-specific Selector residuals."""
        if self.selector_view_residuals is None:
            return self.selectors.new_zeros(())
        return self.selector_view_residuals.float().square().mean()

    def lane_logits(
        self, images: ModelInputs, all_seen_lanes: bool
    ) -> torch.Tensor:
        if self.view_classifier_mode != "shared_post_fusion":
            _, features = self.encode_lanes_with_views(images, all_seen_lanes)
            return self._fuse_view_lane_logits({
                name: self._head_for_view(name)(value)
                for name, value in features.items()
            })
        return self.head(self.encode_lanes(images, all_seen_lanes))

    def current_logits(self, images: ModelInputs) -> torch.Tensor:
        logits = self.current_all_logits(images)
        start = sum(self._task_sizes[: self._current_task_id])
        stop = start + self._task_sizes[self._current_task_id]
        return logits[:, start:stop]

    def current_all_logits(self, images: ModelInputs) -> torch.Tensor:
        return self.lane_logits(images, all_seen_lanes=False)[:, 0]

    def seen_logits(self, images: ModelInputs) -> torch.Tensor:
        lane_ids = self._lane_ids(all_seen_lanes=True)
        logits = self.lane_logits(images, all_seen_lanes=True)
        masks = self.task_class_mask[lane_ids].to(dtype=logits.dtype)
        combined = torch.sum(logits * masks.unsqueeze(0), dim=1)
        return combined[:, : self.seen_classes]

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.base_optimizer_parameters()
        yield from self.adapter_optimizer_parameters()
        yield from self.parax_optimizer_parameters()

    def base_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.representation_optimizer_parameters()
        yield from self.prediction_optimizer_parameters()

    def representation_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        """Shared trainable representation parameters, excluding Adapter."""
        yield self.selectors
        if self.selector_view_residuals is not None:
            yield self.selector_view_residuals
        yield from self.prompts
        yield from self.conditioning_optimizer_parameters()

    def prediction_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        """Classifier and view-fusion parameters optimized by fused DGL loss."""
        yield from self.classifier_optimizer_parameters()
        yield from self.fusion_optimizer_parameters()

    def classifier_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.head.parameters()
        if self.full_view_head is not None:
            yield from self.full_view_head.parameters()
        if self.view_classifier_mode == "private_per_view":
            yield from self.private_view_heads.parameters()

    def conditioning_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        if self.selector_conditioner is not None:
            yield from self.selector_conditioner.active_parameters()

    def fusion_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.view_fusion_module.active_parameters()

    def adapter_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        if self.adapter_bank is not None:
            yield from self.adapter_bank.active_parameters()

    def parax_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        if self.parax_bank is not None:
            yield from self.parax_bank.active_parameters()

    def optimizer_parameter_names(self) -> Tuple[str, ...]:
        names = ["selectors"]
        if self.selector_view_residuals is not None:
            names.append("selector_view_residuals")
        names.extend(f"prompts.{index}" for index in range(len(self.prompts)))
        names.extend(("head.weight", "head.bias"))
        if self.view_classifier_mode == "private_per_view":
            names.extend(
                f"private_view_heads.{index}.{suffix}"
                for index in range(len(self.private_view_heads))
                for suffix in ("weight", "bias")
            )
        if self.selector_conditioner is not None:
            names.extend(
                f"selector_conditioner.{name}"
                for name, parameter in self.selector_conditioner.named_parameters()
                if parameter.requires_grad
            )
        names.extend(
            f"view_fusion_module.{name}"
            for name, parameter in self.view_fusion_module.named_parameters()
            if parameter.requires_grad
        )
        if self.adapter_bank is not None:
            names.extend(
                f"adapter_bank.{name}"
                for name in self.adapter_bank.parameter_names()
                if dict(self.adapter_bank.named_parameters())[name].requires_grad
            )
        if self.parax_bank is not None:
            names.extend(
                f"parax_bank.{name}"
                for name, parameter in self.parax_bank.named_parameters()
                if parameter.requires_grad
            )
        return tuple(names)

    def assert_visual_frozen(self) -> None:
        unexpected = [
            name
            for name, parameter in self.visual_encoder.named_parameters()
            if parameter.requires_grad
        ]
        if unexpected:
            raise RuntimeError(
                "MULTI-LANE visual encoder unexpectedly became trainable: "
                + ", ".join(unexpected)
            )

    def assert_parax_configured(self) -> None:
        if self.parax_mode != "disabled" and self.parax_bank is None:
            raise RuntimeError("ParaX mode is enabled without a ParaX bank")
