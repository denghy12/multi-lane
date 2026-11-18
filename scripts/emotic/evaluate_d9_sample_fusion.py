"""Compare independently learned per-person fusion rules on one protected branch trajectory."""
import argparse
import json
from pathlib import Path

from scripts.emotic.d9_direction_permutation import evaluate as evaluate_permutations
from multi_lane.track_a.d9_relation_fusion import SAMPLE_MODES, sample_readouts
from scripts.emotic.evaluate_d9_protected_gates import evaluate as evaluate_forward
from scripts.emotic.evaluate_d9_relation import write_json


# Unrounded, protected seed1 baseline from d9_protected_fixed/seed1_20261117/fixed/evaluation.
PROTECTED_BASELINE = {
    "val": {"final_mAP": 43.6271956403215, "average_mAP": 51.109441444836264},
    "test": {"final_mAP": 33.201891022384316, "average_mAP": 40.28472011721399},
}
MAP_KEYS = ("final_mAP", "average_mAP")


def comparisons(summary, learned, split, smoke):
    result = {}
    for name in learned:
        scores = summary[name]["all"]
        refs = {"same_branches_fixed": summary["fixed"]["all"],
                "same_branches_equal": summary["equal"]["all"]}
        if not smoke:
            refs["historical_protected_full_train"] = PROTECTED_BASELINE[split]
        result[name] = {}
        for label, ref in refs.items():
            delta = {key: scores[key] - ref[key] for key in MAP_KEYS}
            result[name][label] = dict(delta_mAP_points=delta, both_at_least_reference=min(delta.values()) >= 0)
    if "mid_asl" in learned:
        result["mid_asl_minus_mid_bce"] = {
            key: summary["mid_asl"]["all"][key] - summary["mid_bce"]["all"][key] for key in MAP_KEYS
        }
    return result


def evaluate(root, reference_root, split, batch_size, workers, smoke):
    config = json.loads((root / "config.json").read_text())
    mode = config["mode"]
    if mode not in SAMPLE_MODES:
        raise ValueError(f"Expected from-scratch sample fusion, found {mode}")
    evaluate_forward(root, reference_root, split, batch_size, workers, smoke)
    directory = root / ("smoke_evaluation" if smoke else "evaluation") / split
    path = directory / "result.json"
    report = json.loads(path.read_text())
    learned = [name for name in sample_readouts(mode) if name not in ("fixed", "equal")]
    report.update(mode=mode, learned_readouts=learned, comparisons=comparisons(report["summary"], learned, split, smoke),
                  historical_reference_supplied=reference_root is not None)
    write_json(path, report)
    if split == "test":
        evaluate_permutations(root, smoke, methods=learned, expected_mode=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive; workers must be nonnegative")
    evaluate(args.root, args.reference_root, args.split, args.batch_size, args.workers, args.smoke)


if __name__ == "__main__":
    main()
