"""Sample-level diagnostics for the Selector's image-evidence key/value path."""
from __future__ import annotations

import torch


SELECTOR_EVIDENCE_MODES = ("both", "value_only", "key_only", "none")


def selector_sample_diagnostics(reference_weights, actual_weights, frozen_tokens,
                                adapted_tokens, top_k=10):
    """Inputs describe ONE lane: [B,S,N] weights and [B,N,C] tokens.

    Residual statistics describe the candidate complete Adapter residual even
    when the intervention disconnects it. Energy denominators are sample-wise
    sums over all tokens/channels, shared by the mean and centered components.
    """
    if reference_weights.shape != actual_weights.shape or frozen_tokens.shape != adapted_tokens.shape:
        raise ValueError("Diagnostic tensors differ in shape")
    if reference_weights.ndim != 3 or frozen_tokens.ndim != 3 or top_k <= 0:
        raise ValueError("Invalid Selector diagnostic dimensions/top-k")
    p = actual_weights.detach().float().clamp_min(1e-12)
    q = reference_weights.detach().float().clamp_min(1e-12)
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    k = min(int(top_k), p.shape[-1])
    pi, qi = p.topk(k, dim=-1).indices, q.topk(k, dim=-1).indices
    overlap = (pi.unsqueeze(-1) == qi.unsqueeze(-2)).any(-1).float().mean((-1, -2))
    inputs = frozen_tokens.detach().float()
    delta = adapted_tokens.detach().float() - inputs
    mean = delta.mean(dim=1, keepdim=True)
    centered = delta - mean
    denominator = inputs.square().sum((-1, -2)).clamp_min(1e-12)
    return {
        "attention_kl_current_to_frozen": (p * (p.log()-q.log())).sum(-1).mean(-1).clamp_min(0),
        "attention_kl_frozen_to_current": (q * (q.log()-p.log())).sum(-1).mean(-1).clamp_min(0),
        "attention_top10_overlap": overlap,
        "attention_entropy": -(p*p.log()).sum(-1).mean(-1),
        "attention_max": p.amax(-1).mean(-1),
        "residual_energy_ratio": delta.square().sum((-1, -2)) / denominator,
        "centered_residual_energy_ratio": centered.square().sum((-1, -2)) / denominator,
        "constant_residual_energy_ratio": mean.square().sum((-1, -2)) * delta.shape[1] / denominator,
    }
