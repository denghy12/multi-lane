"""Opt-in two-phase schedule for a single, continuously trained task model.

The return phase replaces the data loader and optionally restarts the LR schedule, preserving the optimizer,
parameters, optimizer moments, scaler, or training history. No default runner
path imports this module unless the experiment explicitly opts in.
"""
from typing import Mapping, Sequence

import torch


def validate_return_training(config: Mapping, epochs: int, update_budget) -> None:
    required = {"initial_epochs", "return_epochs", "lr_factor", "loader"}
    if not required.issubset(config):
        raise ValueError("Return training requires initial_epochs/return_epochs/lr_factor/loader")
    initial, returned = int(config["initial_epochs"]), int(config["return_epochs"])
    if initial <= 0 or returned <= 0 or initial + returned != epochs:
        raise ValueError("Return phases must be positive and sum to the total epoch budget")
    if update_budget is not None:
        raise ValueError("Return training requires explicit phase epochs, not a fixed-update cap")
    if not 0 < float(config["lr_factor"]) <= 1:
        raise ValueError("Return LR factor must be in (0, 1]")
    if not config.get("restart_scheduler", True) and config["lr_factor"] != 1:
        raise ValueError("Continuous cosine requires a unit return LR factor")
    if config["loader"] is None:
        raise ValueError("Return training needs a complete legal TRAIN loader")


def reset_return_scheduler(optimizer, peak_lrs: Sequence[float], config: Mapping):
    """Reset only the LR schedule, retaining the very same Adam state objects."""
    if len(peak_lrs) != len(optimizer.param_groups):
        raise ValueError("Original LR peaks must match every optimizer parameter group")
    for group, peak in zip(optimizer.param_groups, peak_lrs):
        rate = float(peak) * float(config["lr_factor"])
        # CosineAnnealingLR consults initial_lr when constructed after another
        # scheduler. Updating only lr would silently keep the old peak.
        group["lr"] = rate
        group["initial_lr"] = rate
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["return_epochs"])
    )
