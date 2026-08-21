"""GPU forward/backward smoke for the strict Track-A model."""

from __future__ import annotations

import argparse

import torch

from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .runner import (
    backward_routed_training_losses,
    compute_asymmetric_training_loss,
    compute_training_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument(
        "--adapter-mode",
        choices=("disabled", "task_lane", "image_token"),
        default="disabled",
    )
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=64)
    parser.add_argument(
        "--adapter-layer-indices",
        type=int,
        nargs="+",
        default=(11,),
        help="Zero-based CLIP transformer block indices used by adapters.",
    )
    parser.add_argument(
        "--adapter-task-init",
        choices=("independent", "copy_previous"),
        default="independent",
    )
    parser.add_argument(
        "--loss-routing",
        choices=("joint_bce", "model_asl", "adapter_asl", "both_asl"),
        default="joint_bce",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Run the smoke in full precision, matching stable search runs.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    amp = not args.no_amp
    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = MultiLaneModel(
        visual,
        (5, 3, 3, 3, 3, 3, 3, 3),
        adapter_mode=args.adapter_mode,
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        adapter_layer_indices=tuple(args.adapter_layer_indices),
        adapter_residual_scale=0.1,
        adapter_task_initialization=args.adapter_task_init,
    ).float().cuda()
    model.activate_task(0)
    images = torch.randn(2, 3, 224, 224, device="cuda")
    if model.adapter_bank is not None:
        for layer_index in args.adapter_layer_indices:
            if args.adapter_mode == "image_token":
                probe = torch.ones(
                    1, 5, model.width, device="cuda", dtype=torch.float32
                )
                adapted_probe = model.adapter_bank.adapted_tokens_for_layer(
                    layer_index, probe, (0,)
                )
                adapter_delta = adapted_probe - probe.unsqueeze(0)
            else:
                probe = torch.ones(
                    1, 1, 1, model.width, device="cuda", dtype=torch.float32
                )
                adapter_delta = model.adapter_bank.delta_for_layer(
                    layer_index, probe, (0,)
                )
            if torch.count_nonzero(adapter_delta).item() != 0:
                raise RuntimeError(
                    "Zero-initialized adapter produced a nonzero residual at "
                    f"layer {layer_index}"
                )
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            adapter_logits = model.current_all_logits(images)
            model.set_adapter_runtime_enabled(False)
            baseline_logits = model.current_all_logits(images)
            model.set_adapter_runtime_enabled(True)
        max_initial_difference = float(
            (adapter_logits - baseline_logits).abs().max().cpu()
        )
        tolerance = 1e-4 if amp else 1e-6
        if not torch.allclose(
            adapter_logits, baseline_logits, atol=tolerance, rtol=tolerance
        ):
            raise RuntimeError(
                "Zero-initialized adapter changed initial logits beyond "
                f"{'AMP' if amp else 'FP32'} "
                f"tolerance: max_difference={max_initial_difference}"
            )
    model.train()
    model_parameters = list(model.base_optimizer_parameters())
    adapter_parameters = list(model.adapter_optimizer_parameters())
    if args.loss_routing in {"adapter_asl", "both_asl"} and not adapter_parameters:
        raise RuntimeError("Adapter ASL smoke requires an enabled Adapter")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    with torch.cuda.amp.autocast(enabled=amp):
        logits = model.current_all_logits(images)
        current_targets = torch.zeros(2, 5, device="cuda")
        bce_loss = compute_training_loss(
            logits, current_targets, range(5), 1.0, "legacy_full_zero"
        )
        asl_loss = compute_asymmetric_training_loss(
            logits, current_targets, range(5), 1.0, "legacy_full_zero"
        )
        model_loss = (
            asl_loss
            if args.loss_routing in {"model_asl", "both_asl"} else bce_loss
        )
        adapter_loss = (
            asl_loss
            if args.loss_routing in {"adapter_asl", "both_asl"} else bce_loss
        )
    backward_routed_training_losses(
        model_loss=model_loss,
        adapter_loss=adapter_loss,
        model_parameters=model_parameters,
        adapter_parameters=adapter_parameters,
        scaler=scaler,
        same_objective=args.loss_routing in {"joint_bce", "both_asl"},
    )
    model.assert_visual_frozen()
    if model.selectors.grad is None or not torch.isfinite(model.selectors.grad).all():
        raise RuntimeError("Selector gradient smoke failed")
    if adapter_parameters and (
        any(parameter.grad is None for parameter in adapter_parameters)
        or any(
            not torch.isfinite(parameter.grad).all()
            for parameter in adapter_parameters
        )
    ):
        raise RuntimeError("Adapter gradient smoke failed")
    if any(parameter.grad is not None for parameter in model.visual_encoder.parameters()):
        raise RuntimeError("Frozen visual tower received gradients")
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        seen = model.seen_logits(images)
    if tuple(seen.shape) != (2, 5) or not torch.isfinite(seen).all():
        raise RuntimeError("Concat inference smoke failed")
    expected = 689178 + (
        model.adapter_bank.per_task_parameter_count()
        if model.adapter_bank is not None else 0
    )
    trainable = sum(p.numel() for p in model.optimizer_parameters())
    if trainable != expected:
        raise RuntimeError(f"Expected {expected} trainable parameters, got {trainable}")
    print(
        "MULTI_LANE_TRACK_A_SMOKE_OK "
        f"adapter_mode={args.adapter_mode} task_init={args.adapter_task_init} "
        f"loss_routing={args.loss_routing} "
        f"precision={'amp' if amp else 'fp32'} "
        f"adapter_layers={','.join(map(str, args.adapter_layer_indices))} "
        f"trainable_parameters={trainable} "
        f"max_initial_difference={max_initial_difference if model.adapter_bank is not None else 0.0}"
    )


if __name__ == "__main__":
    main()
