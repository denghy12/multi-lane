"""Same-batch TRAIN counterfactuals for an auxiliary Adam update."""
import torch


def measure_candidate_effect(*, model, optimizer, baseline_parameters, images,
                             targets, current, temperature, loss_mode, amp):
    """Compare selected parameters with this step's original-B0 candidate.

    Only transient parameter copies and no-grad forwards are used. The
    selected real Adam state is never changed; all RNG/training flags and
    parameter values are restored even if a diagnostic forward fails.
    Call after normal per-batch diagnostic collection because the model's
    transient forward caches are refreshed by these diagnostic forwards.
    """
    from .runner import compute_training_loss
    from .task_recovery import rng_state, restore_rng

    selected = {p: p.detach().clone() for group in optimizer.param_groups
                for p in group["params"]}
    if set(selected) != set(baseline_parameters):
        raise ValueError("Counterfactual must cover every optimizer parameter")
    flags = [(module, module.training) for module in model.modules()]
    rng = rng_state()

    def evaluate():
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            logits, views, _ = model.current_all_logits_with_view_features(images)
            return {
                "full": float(compute_training_loss(
                    views["full"], targets, current, temperature, loss_mode)),
                "fused": float(compute_training_loss(
                    logits, targets, current, temperature, loss_mode)),
            }

    try:
        model.eval()
        with torch.no_grad():
            for p, candidate in baseline_parameters.items():
                p.copy_(candidate)
        base = evaluate()
        with torch.no_grad():
            for p, value in selected.items():
                p.copy_(value)
        actual = evaluate()
        return {**{name + "_bce_extra": actual[name] - base[name] for name in base},
                **{name + "_base_bce": value for name, value in base.items()},
                **{name + "_selected_bce": value for name, value in actual.items()}}
    finally:
        with torch.no_grad():
            for p, value in selected.items():
                p.copy_(value)
        for module, training in flags:
            module.training = training
        restore_rng(rng)

