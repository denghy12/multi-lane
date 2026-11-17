"""Detached finite-transfer targets and a fixed update budget for D9 gates."""
import math

import torch
import torch.nn.functional as F


def gate_update_due(batch_index, batches_per_epoch, epoch, epochs, budget):
    """Uniform integer schedule; epoch is one-based, batch_index zero-based."""
    total = batches_per_epoch * epochs
    if not 0 < budget <= total:
        raise ValueError('Gate update budget exceeds the branch batches')
    if not 0 <= batch_index < batches_per_epoch or not 1 <= epoch <= epochs:
        raise ValueError('Gate schedule position is out of range')
    position = (epoch - 1) * batches_per_epoch + batch_index
    return (position + 1) * budget // total > position * budget // total


@torch.no_grad()
def direction_benefits(branches, labels, reliable, transfer=.05, base_weights=None):
    """Return fixed-center BCE reduction and usable [Person, Face] targets."""
    if branches.shape != (*labels.shape, 3) or labels.ndim != 2:
        raise ValueError('Expected branch logits [people, current classes, views]')
    if not 0 < transfer < .64:
        raise ValueError('Transfer must preserve positive Full weight')
    branches, labels = branches.detach().float(), labels.detach().float()
    reliable = reliable.reshape(-1).bool()
    if len(reliable) != len(labels):
        raise ValueError('Reliability and people differ')
    odds = branches.new_tensor([0., math.log(.16 / .64), math.log(.20 / .64)])
    odds = odds.expand(len(labels), 3).clone()
    odds[:, 2].masked_fill_(~reliable, -torch.inf)
    weights = odds.softmax(-1) if base_weights is None else base_weights.detach().float()
    if weights.shape != (len(labels), 3) or not torch.isfinite(weights).all():
        raise ValueError('Invalid direction reference weights')
    if (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=2e-6):
        raise ValueError('Direction reference must lie on the view simplex')
    if weights[~reliable, 2].count_nonzero():
        raise ValueError('Invalid Face weight in direction reference')
    face = torch.where(reliable[:, None], branches[..., 2], 0.)
    logits = (weights[:, 0:1] * branches[..., 0]
              + weights[:, 1:2] * branches[..., 1] + weights[:, 2:3] * face)
    base_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').mean(1)
    reductions = []
    amount = weights[:, :1].clamp(max=transfer)
    for other in (branches[..., 1], face):
        perturbed = logits + amount * (other - branches[..., 0])
        loss = F.binary_cross_entropy_with_logits(perturbed, labels, reduction='none').mean(1)
        reductions.append(base_loss - loss)
    benefit = torch.stack(reductions, 1)
    valid = torch.stack((torch.ones_like(reliable), reliable), 1)
    benefit = torch.where(valid, benefit, 0.)
    if not torch.isfinite(benefit).all():
        raise FloatingPointError('Nonfinite direction target')
    return benefit, valid


def deranged_targets(target, valid, reliable, seed):
    """Preserve target counts, and exclude singleton groups symmetrically."""
    reliable = reliable.reshape(-1).bool()
    generator = torch.Generator().manual_seed(seed)
    donors = torch.arange(len(target), device=target.device)
    usable = valid.clone()
    for condition in (False, True):
        rows = torch.where(reliable == condition)[0]
        if len(rows) < 2:
            usable[rows] = False
            continue
        order = torch.randperm(len(rows), generator=generator).to(rows.device)
        shuffled = rows[order]
        donors[shuffled] = shuffled.roll(1)
    if not usable.any():
        raise ValueError('No nonsingleton reliability cohort for matched direction training')
    return target[donors], usable, donors
