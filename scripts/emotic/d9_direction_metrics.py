"""Read-only evaluation of person-specific auxiliary-view benefit prediction."""
import numpy as np
import torch

from multi_lane.track_a.d9_direction_objective import direction_benefits


TRANSFER = 0.05


def tied_auc(scores, targets):
    """Positive/negative ranking probability, awarding half credit to exact ties."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUC requires aligned finite one-dimensional inputs")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ranked = scores[order]
    # Equal-score intervals receive their average one-based rank.
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    average_ranks = (starts + 1 + ends) / 2
    ranks = np.repeat(average_ranks, ends - starts)
    positive_ranks = float(ranks[targets[order]].sum())
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def selection_statistics(scores, reduction):
    scores = np.asarray(scores, dtype=np.float64)
    reduction = np.asarray(reduction, dtype=np.float64)
    if scores.shape != reduction.shape or scores.ndim != 1:
        raise ValueError("Direction scores and BCE reductions must be aligned vectors")
    if not np.isfinite(scores).all() or not np.isfinite(reduction).all():
        raise ValueError("Direction diagnostic contains nonfinite values")
    labels = reduction > 0
    chosen = scores > 0
    n = len(labels)
    positives, selected = int(labels.sum()), int(chosen.sum())
    return dict(
        people=n, improvement_positive=positives, no_improvement=n - positives,
        exact_zero_reduction=int((reduction == 0).sum()),
        prevalence=float(labels.mean()) if n else None,
        auc=tied_auc(scores, labels),
        auc_status="defined" if 0 < positives < n else "requires_positive_and_negative",
        predicted_positive=selected,
        selected_fraction=float(chosen.mean()) if n else None,
        selected_precision=float(labels[chosen].mean()) if selected else None,
        threshold_zero_accuracy=float((chosen == labels).mean()) if n else None,
        mean_bce_reduction=float(reduction.mean()) if n else None,
        selected_mean_bce_reduction=float(reduction[chosen].mean()) if selected else None,
        remaining_mean_bce_reduction=float(reduction[~chosen].mean()) if n > selected else None,
    )


def direction_metrics(targets, reliable, branches, predictions, task_sizes, center_weights=None):
    """Diagnose raw direction scores within each task and Face availability cohort.

    A/B/C labels describe separate finite perturbations at the fixed center.
    Center_b labels use the concurrently learned center-only weights.
    Person and Face are perturbed independently.
    Selection thresholds are fixed at zero; no evaluation-label tuning is performed.
    """
    sizes = list(task_sizes)
    expected = (len(targets), len(sizes), 2)
    if targets.ndim != 2 or targets.shape[1] != sum(sizes):
        raise ValueError("Targets and task partition disagree")
    if branches.shape != (*targets.shape, 3) or reliable.shape != (len(targets),):
        raise ValueError("Branches and reliability must align with targets")
    if not {"a", "b", "c"}.issubset(predictions):
        raise ValueError("A, B and C direction scores are required")
    if "center_b" in predictions and (center_weights is None or tuple(center_weights.shape) != (*expected[:2], 3)):
        raise ValueError("center_b diagnostics require aligned learned center weights")
    reductions, validity, learned_reductions, learned_validity = [], [], [], []
    start = 0
    for size in sizes:
        columns = slice(start, start + size)
        decrease, valid = direction_benefits(branches[:, columns], targets[:, columns], reliable,
                                             transfer=TRANSFER)
        reductions.append(decrease.detach().cpu())
        validity.append(valid.detach().cpu())
        if "center_b" in predictions:
            gain, usable = direction_benefits(branches[:, columns], targets[:, columns], reliable,
                                               transfer=TRANSFER, base_weights=center_weights[:, len(reductions) - 1])
            learned_reductions.append(gain.detach().cpu())
            learned_validity.append(usable.detach().cpu())
        start += size
    reductions = torch.stack(reductions, dim=1).numpy()
    validity = torch.stack(validity, dim=1).numpy()
    reliable_array = reliable.detach().cpu().bool().numpy()
    arrays = dict(direction_bce_reduction=reductions, direction_valid=validity)
    if learned_reductions:
        arrays["center_direction_bce_reduction"] = torch.stack(learned_reductions, dim=1).numpy()
        arrays["center_direction_valid"] = torch.stack(learned_validity, dim=1).numpy()
    methods = {}
    for name, tensor in predictions.items():
        if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid raw direction scores for {name}")
        gains = arrays["center_direction_bce_reduction"] if name == "center_b" else reductions
        valid = arrays["center_direction_valid"] if name == "center_b" else validity
        values = tensor.detach().cpu().numpy()
        arrays[name + "_direction_scores"] = values
        task_reports = []
        for task in range(len(sizes)):
            groups = {}
            for cohort, people in (("unreliable", ~reliable_array), ("reliable", reliable_array)):
                groups[cohort] = {}
                for direction, view in enumerate(("person", "face")):
                    keep = people & valid[:, task, direction]
                    groups[cohort][view] = selection_statistics(values[keep, task, direction],
                                                                gains[keep, task, direction])
            task_reports.append(dict(task=task, groups=groups))
        methods[name] = task_reports
    protocol = dict(
        transfer_from_full=TRANSFER,
        center="A/B/C fixed R1 weights; center_b uses learned center-only weights from the same checkpoint",
        learned_center_transfer="min(0.05, current Full weight), per person",
        target="Per-person BCE over this task's classes before minus after independent finite transfer",
        positive="BCE reduction > 0; exact zero is no improvement",
        score="Raw learned log-odds correction; threshold zero is prespecified",
        groups="Each task head separately, within Face reliability cohort; unavailable Face excluded",
        auc="Tie-averaged ranks; null when positives or negatives are absent",
        use="Read-only diagnosis; neither thresholds nor model weights fitted to Val/Test labels",
    )
    return dict(protocol=protocol, methods=methods), arrays
