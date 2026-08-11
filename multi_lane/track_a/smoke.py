"""GPU forward/backward smoke for the strict Track-A model."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-checkpoint", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = MultiLaneModel(visual, (5, 3, 3, 3, 3, 3, 3, 3)).float().cuda()
    model.activate_task(0)
    model.train()
    images = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.cuda.amp.autocast():
        logits = model.current_all_logits(images)
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), torch.zeros_like(logits)
        )
    loss.backward()
    model.assert_visual_frozen()
    if model.selectors.grad is None or not torch.isfinite(model.selectors.grad).all():
        raise RuntimeError("Selector gradient smoke failed")
    if any(parameter.grad is not None for parameter in model.visual_encoder.parameters()):
        raise RuntimeError("Frozen visual tower received gradients")
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        seen = model.seen_logits(images)
    if tuple(seen.shape) != (2, 5) or not torch.isfinite(seen).all():
        raise RuntimeError("Concat inference smoke failed")
    expected = 689178
    trainable = sum(p.numel() for p in model.optimizer_parameters())
    if trainable != expected:
        raise RuntimeError(f"Expected {expected} trainable parameters, got {trainable}")
    print(f"MULTI_LANE_TRACK_A_SMOKE_OK trainable_parameters={trainable}")


if __name__ == "__main__":
    main()
