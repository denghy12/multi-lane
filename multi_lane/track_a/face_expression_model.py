"""Incremental Face expert over a frozen facial-expression encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .adapter import TransformerBlockAdapter


def load_frozen_emotieff_encoder(checkpoint: Path) -> Tuple[nn.Module, int]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing EmotiEffLib checkpoint: {checkpoint}")
    # The official artifact serializes the complete timm EfficientNet object.
    # Importing timm registers its module classes before torch unpickles it.
    import timm  # noqa: F401

    try:
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.0 does not expose weights_only.
        model = torch.load(checkpoint, map_location="cpu")
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        linear_layers = [module for module in classifier if isinstance(module, nn.Linear)]
        if len(linear_layers) != 1:
            raise TypeError("Unsupported EmotiEffLib sequential classifier")
        classifier = linear_layers[0]
    if not isinstance(classifier, nn.Linear) or classifier.out_features not in (7, 8, 10):
        raise TypeError("Checkpoint is not a supported EmotiEffLib emotion model")
    feature_dim = int(classifier.in_features)
    model.classifier = nn.Identity()
    model.requires_grad_(False)
    model.eval()
    return model, feature_dim


class TaskFeatureAdapterBank(nn.Module):
    def __init__(
        self,
        task_count: int,
        feature_dim: int,
        bottleneck_dim: int,
        residual_scale: float,
        activation: str,
    ) -> None:
        super().__init__()
        if task_count <= 0 or feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Face feature Adapter dimensions must be positive")
        if not 0 < residual_scale <= 1:
            raise ValueError("Face feature Adapter residual scale must be in (0, 1]")
        self.adapters = nn.ModuleList(
            TransformerBlockAdapter(feature_dim, bottleneck_dim, activation)
            for _ in range(task_count)
        )
        self.residual_scale = float(residual_scale)
        self._current_task_id = -1
        self.requires_grad_(False)

    def activate_task(self, task_id: int) -> None:
        if not 0 <= task_id < len(self.adapters):
            raise ValueError("Face feature Adapter task id is outside the protocol")
        self.requires_grad_(False)
        self.adapters[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        return features + self.residual_scale * self.adapters[task_id](features)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id < 0:
            return iter(())
        return iter(self.adapters[self._current_task_id].parameters())

    def per_task_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapters[0].parameters())


class FaceExpressionIncrementalModel(nn.Module):
    """Frozen expression features with independent, frozen-after-task heads."""

    VARIANTS = ("projection", "bottleneck_adapter")

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        task_sizes: Sequence[int],
        variant: str,
        bottleneck_dim: int = 32,
        residual_scale: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown Face expression variant: {variant}")
        if feature_dim <= 0 or not task_sizes or any(int(size) <= 0 for size in task_sizes):
            raise ValueError("Face expression model dimensions must be positive")
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.feature_dim = int(feature_dim)
        self.task_sizes = tuple(int(size) for size in task_sizes)
        self.variant = variant
        self.heads = nn.ModuleList(
            nn.Linear(self.feature_dim, task_size) for task_size in self.task_sizes
        )
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
        self.adapter_bank: Optional[TaskFeatureAdapterBank] = None
        if variant == "bottleneck_adapter":
            with torch.random.fork_rng(devices=[]):
                self.adapter_bank = TaskFeatureAdapterBank(
                    len(self.task_sizes), self.feature_dim, bottleneck_dim,
                    residual_scale, activation,
                )
        self._current_task_id = -1
        self.heads.requires_grad_(False)

    @property
    def current_task_id(self) -> int:
        return self._current_task_id

    @property
    def seen_classes(self) -> int:
        return sum(self.task_sizes[: self._current_task_id + 1])

    def activate_task(self, task_id: int) -> None:
        if task_id != self._current_task_id + 1:
            raise RuntimeError("Face expression tasks must be activated sequentially")
        if not 0 <= task_id < len(self.task_sizes):
            raise ValueError("Face expression task id is outside the protocol")
        self.heads.requires_grad_(False)
        self.heads[task_id].requires_grad_(True)
        if self.adapter_bank is not None:
            self.adapter_bank.activate_task(task_id)
        self._current_task_id = int(task_id)

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm statistics in the pretrained expression encoder are part
        # of the frozen representation and must never drift across tasks.
        self.encoder.eval()
        return self

    def frozen_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if features.shape[1] != self.feature_dim or not torch.isfinite(features).all():
            raise RuntimeError("Frozen expression encoder returned invalid features")
        return features.float()

    def task_features(self, images: torch.Tensor, task_id: int) -> torch.Tensor:
        features = self.frozen_features(images)
        if self.adapter_bank is not None:
            features = self.adapter_bank(features, task_id)
        return features

    def current_task_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self._current_task_id < 0:
            raise RuntimeError("No Face expression task is active")
        features = self.task_features(images, self._current_task_id)
        return self.heads[self._current_task_id](features)

    def current_all_logits(self, images: torch.Tensor) -> torch.Tensor:
        current = self.current_task_logits(images)
        start = sum(self.task_sizes[: self._current_task_id])
        stop = start + self.task_sizes[self._current_task_id]
        return torch.cat(
            [
                current.new_zeros((len(current), start)),
                current,
                current.new_zeros((len(current), sum(self.task_sizes) - stop)),
            ],
            dim=1,
        )

    def current_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.current_task_logits(images)

    def seen_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self._current_task_id < 0:
            raise RuntimeError("No Face expression task is active")
        rows = []
        frozen = self.frozen_features(images)
        for task_id in range(self._current_task_id + 1):
            features = frozen
            if self.adapter_bank is not None:
                features = self.adapter_bank(features, task_id)
            rows.append(self.heads[task_id](features))
        return torch.cat(rows, dim=1)

    def base_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id < 0:
            return iter(())
        return iter(self.heads[self._current_task_id].parameters())

    def adapter_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        if self.adapter_bank is None:
            return iter(())
        return self.adapter_bank.active_parameters()

    def assert_encoder_frozen(self) -> None:
        names = [name for name, value in self.encoder.named_parameters() if value.requires_grad]
        if names:
            raise RuntimeError("Expression encoder unexpectedly became trainable: " + ", ".join(names))
