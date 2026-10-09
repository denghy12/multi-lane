"""CPU-only analysis of saved joint26 validation predictions; no model training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare_joint26 import METHODS, compare, require
from .runner import CLASS_ORDER, average_precision


def map_score(scores, targets):
    return float(np.mean([100 * average_precision(scores[:, c], targets[:, c]) for c in range(26)]))


def reduced_view_logits(full, person, face, reliable, keep_person=True, keep_face=True):
    """Renormalize the existing fixed prior; never fit weights on validation."""
    weights = np.tile([0.64, 0.16, 0.20], (len(full), 1)).astype(np.float32)
    weights[~reliable] = [0.8, 0.2, 0]
    if not keep_person:
        weights[:, 1] = 0
    if not keep_face:
        weights[:, 2] = 0
    weights /= weights.sum(axis=1, keepdims=True)
    return sum(value * weights[:, i:i+1] for i, value in enumerate((full, person, face)))


def analyze(batch_root: Path, bootstrap_replicates=500):
    comparison = compare(batch_root, allow_skipped_updates=True)
    scores, targets, ids = {}, None, None
    views = {}
    for method in METHODS:
        with np.load(batch_root / method / "val_scores" / "task0.npz") as z:
            scores[method] = z["probabilities"].copy()
            targets, ids = z["targets"].copy(), z["sample_ids"].copy()
        if not method.startswith("THREE"):
            continue
        with np.load(batch_root / method / "view_val_scores" / "task0.npz") as z:
            require(np.array_equal(ids, z["sample_ids"]) and np.array_equal(targets, z["targets"]), "View scores mismatch")
            full, person, face, reliable = [z[k] for k in ("full_logits", "person_logits", "face_logits", "face_reliable")]
            reconstructed = reduced_view_logits(full, person, face, reliable)
            max_difference = float(np.max(np.abs(reconstructed - z["fused_logits"])))
            reconstructed_map = map_score(reconstructed, targets)
            require(abs(reconstructed_map - comparison["runs"][method]["mAP"]) < 0.1,
                    "Float32 logit reconstruction does not match fixed feature fusion")
            conditions = {
                "同一三路模型，仅保留 Full": full,
                "同一三路模型，移除 Face，Full+Person 权重重归一": reduced_view_logits(full, person, face, reliable, keep_face=False),
                "同一三路模型，移除 Person，Full+Face 权重重归一": reduced_view_logits(full, person, face, reliable, keep_person=False),
                "同一三路模型，三路固定融合重建": reconstructed,
            }
            maps = {name: map_score(value, targets) for name, value in conditions.items()}
            values = list(maps.values())
            views[method] = {"conditions_mAP": maps, "max_reconstruction_logit_difference": max_difference,
                             "face_increment_over_full_person": values[3]-values[1],
                             "person_increment_over_full_face": values[3]-values[2],
                             "note": "Inference-only removal from an already three-view-trained model. Float32 affine reconstruction differs slightly from AMP feature fusion; not separately trained two-view ablations."}
    groups = {}
    for index, sample_id in enumerate(ids):
        groups.setdefault(str(sample_id).rsplit("#person=", 1)[0], []).append(index)
    group_indices = list(groups.values())
    rng = np.random.default_rng(20261009)
    pairs = {
        "三路减 Full（无 ParaX）": ("THREE_VIEW", "FULL_ONLY"),
        "三路 ParaX 减三路基线": ("THREE_VIEW_PARAX", "THREE_VIEW"),
        "Full ParaX 减 Full 基线": ("FULL_ONLY_PARAX", "FULL_ONLY"),
    }
    samples = {name: [] for name in pairs}
    for _ in range(bootstrap_replicates):
        chosen = rng.integers(0, len(group_indices), len(group_indices))
        rows = np.concatenate([group_indices[i] for i in chosen])
        maps = {method: map_score(value[rows], targets[rows]) for method, value in scores.items()}
        for name, (a, b) in pairs.items():
            samples[name].append(maps[a]-maps[b])
    intervals = {name: {"observed_difference": map_score(scores[a], targets)-map_score(scores[b], targets),
                        "percentile_95_interval": [float(v) for v in np.percentile(samples[name], [2.5, 97.5])]}
                 for name, (a, b) in pairs.items()}
    class_differences = {}
    for name, (a, b) in pairs.items():
        ap_a, ap_b = comparison["runs"][a]["per_class_ap"], comparison["runs"][b]["per_class_ap"]
        class_differences[name] = {cls: float(x-y) for cls, x, y in zip(CLASS_ORDER, ap_a, ap_b)}
    result = {"view_removal": views, "class_differences": class_differences,
              "image_group_bootstrap": {"replicates": bootstrap_replicates, "groups": len(groups), "differences": intervals,
                                         "note": "Validation-image sampling uncertainty conditional on these four trained seed0 models; not training-seed uncertainty or a guarantee of test generalization."}}
    (batch_root / "mechanism_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"view_removal": views, "bootstrap": result["image_group_bootstrap"],
                      "class_counts_improved": {name: sum(value > 0 for value in data.values()) for name, data in class_differences.items()}},
                     indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    analyze(args.batch_root, args.bootstrap_replicates)
