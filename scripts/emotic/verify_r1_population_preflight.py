"""Exact read-only replay gate for paired-population versus original return smoke."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a import runner
from multi_lane.track_a.fusion_population import METHODS, in_heldout_group
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source
from scripts.emotic.run_r1_single_trajectory_fusion import VIEWS, assert_same, branch_paths

CURRENT = Path("output/r1_fusion_population_train/smoke01_20261113")
REFERENCE = Path("../multi-lane-main-r1-fusion-return-train-20261113/output/r1_fusion_return_train/smoke01_20261113/holdout_return")


def load_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def sources(current, reference):
    paths = []
    for root in (current, reference):
        paths += [root/name for name in ("completion.json", "training_budget_checks.json", "result.json")]
        for view, branch in branch_paths(root).items():
            paths.append(branch/"config.json")
            for task in range(2):
                paths += [branch/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth" for epoch in (1, 2, 3)]
                paths.append(root/"observations"/f"task{task}"/f"{view}.npz")
                if root == current:
                    paths += [root/"population_observations"/f"task{task}"/f"{view}.npz",
                              root/"population_manifests"/view/f"task{task}.json"]
        for task in range(2):
            paths += [root/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth" for epoch in (1, 2, 3)]
            for split in ("val", "test"):
                for phase in ("pre_return", "final"):
                    paths.append(root/"predictions"/phase/split/f"task{task}.npz")
    return paths


def verify(current, reference, output):
    torch.set_num_threads(2)
    result = {"status": "running", "current": str(current), "reference": str(reference),
              "gradient_updates": 0, "tensor_tolerance": 0, "ignored":
              ["config/git/path provenance", "training history timing and device memory telemetry"]}
    hashes = {}
    try:
        required = sources(current, reference)
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete smoke outputs: {missing[:5]}")
        hashes = {str(path): sha(path) for path in required}
        for root in (current, reference):
            done = read(root/"completion.json")
            if done.get("status") != "complete" or done.get("smoke") is not True or done.get("validation_parameter_updates") != 0:
                raise AssertionError(f"Not a complete no-Val-fit smoke: {root}")
        branch_rows = []
        for view in VIEWS:
            for task in range(2):
                for epoch in (1, 2, 3):
                    states = [torch.load(branch_paths(root)[view]/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth",
                                         map_location="cpu") for root in (current, reference)]
                    a, b = states; prefix = f"{view}/task{task}/epoch{epoch}"
                    for key in ("task", "model", "rng"):
                        assert_same(a[key], b[key], prefix+"/"+key)
                    keys = ("optimizer", "scheduler", "scaler", "next_epoch", "completed_task_updates", "optimizer_parameter_names")
                    for key in keys:
                        assert_same(a["training"][key], b["training"][key], prefix+"/training/"+key)
                    # Compare learning diagnostics while ignoring performance telemetry only.
                    history = []
                    for state in states:
                        history.append([{k: v for k, v in row.items()
                                         if k != "elapsed_seconds" and not k.startswith("cuda_peak_")}
                                        for row in state["training"]["history"]])
                    assert_same(*history, prefix+"/learning_history")
                    if any(row["skipped_optimizer_steps"] for row in history[0]):
                        raise AssertionError(prefix+"/skipped_updates")
                    branch_rows.append({"view": view, "task": task, "epoch": epoch,
                                        "model_optimizer_scheduler_scaler_rng_exact": True})
        result["branch_epochs"] = branch_rows
        assert len(branch_rows) == 18

        source = load_source(branch_paths(current)["person"], input_paths(), "train")
        lookup = {str(name): i for i, name in enumerate(source.sample_ids)}
        observation_rows = []
        for task in range(2):
            classes, seen = set(runner.task_indices(task)), set(runner.seen_indices(task))
            for view in VIEWS:
                new = load_npz(current/"observations"/f"task{task}"/f"{view}.npz")
                old = load_npz(reference/"observations"/f"task{task}"/f"{view}.npz")
                pop = load_npz(current/"population_observations"/f"task{task}"/f"{view}.npz")
                assert_same(new, old, f"original_observations/{view}/task{task}")
                n = len(new["ids"])
                for key in ("ids", "logits", "targets", "reliable"):
                    assert_same(new[key], pop[key][:n], f"population_prefix/{view}/{task}/{key}")
                if (len(set(pop["ids"])) != len(pop["ids"])
                        or not np.all(new["targets"].sum(axis=1) > 0)
                        or not np.all(pop["targets"][n:] == 0)):
                    raise AssertionError(f"Invalid population membership: {view}/{task}")
                manifest = read(current/"population_manifests"/view/f"task{task}.json")
                assert_same(list(pop["ids"]), manifest["ids"]["seen_population"], f"manifest/ids/{view}/{task}")
                expected = [str(source.sample_ids[i]) for i, target in enumerate(source.targets)
                            if set(target) & seen and in_heldout_group(source.sample_ids[i])]
                if set(pop["ids"]) != set(expected):
                    raise AssertionError(f"Seen-H set excludes legal or contains illegal people: {view}/{task}")
                for name in pop["ids"][n:]:
                    target = set(source.targets[lookup[str(name)]])
                    if not target & seen or target & classes:
                        raise AssertionError(f"Future-only/current-positive entered added population: {name}")
                added = len(pop["ids"])-n
                if (task == 0 and added != 0) or (task == 1 and added <= 0):
                    raise AssertionError(f"Wrong task0/task1 negative coverage: {view}/{task}/{added}")
                if manifest["counts"]["added_negative"] != added:
                    raise AssertionError(f"Negative count differs from manifest: {view}/{task}")
                observation_rows.append({"task": task, "view": view, "original_people": n,
                                         "added_negative_people": added, "future_only_excluded": True,
                                         "original_observations_exact": True})
        result["observation_checks"] = observation_rows
        fusion_rows = []
        for task in range(2):
            for epoch in (1, 2, 3):
                a = torch.load(current/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
                b = torch.load(reference/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
                bank = a["fusion"]["banks"]["current_positive"]
                assert_same(bank, b["fusion"], f"fusion/current_positive/task{task}/epoch{epoch}")
                if task == 0:
                    assert_same(bank, a["fusion"]["banks"]["seen_population"], f"fusion/task0_banks/epoch{epoch}")
                if epoch >= 2:
                    if any(a["fusion"]["banks"][m]["updates"][task] != 2 for m in METHODS):
                        raise AssertionError(f"Smoke fusion budget mismatch: {task}/{epoch}")
                fusion_rows.append({"task": task, "epoch": epoch, "original_fusion_state_exact": True})
        result["fusion_checks"] = fusion_rows
        forward_rows = []
        for split in ("val", "test"):
            for phase in ("pre_return", "final"):
                for task in range(2):
                    a = load_npz(current/"predictions"/phase/split/f"task{task}.npz")
                    b = load_npz(reference/"predictions"/phase/split/f"task{task}.npz")
                    for key in ("ids", "targets", "reliable", "branch_logits", "fixed"):
                        assert_same(a[key], b[key], f"forward/{split}/{phase}/{task}/{key}")
                    assert_same(a["current_positive"], b["learned"], f"forward/{split}/{phase}/{task}/original_fusion")
                    if task == 0:
                        assert_same(a["current_positive"], a["seen_population"], f"forward/{split}/{phase}/{task}/identical_task0")
                    forward_rows.append({"split": split, "phase": phase, "task": task, "source_readout_exact": True})
        result["whole_model_forward_checks"] = forward_rows
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["first_error"] = repr(error)
        print("PREFLIGHT_FAILED", repr(error), flush=True)
        raise
    finally:
        unchanged = all(Path(path).is_file() and sha(Path(path)) == digest for path, digest in hashes.items())
        result["sources_sha_unchanged"] = unchanged
        result["source_sha256"] = hashes
        if not unchanged:
            result["status"] = "failed"
            result["source_change_detected"] = True
        write(output, result)
        if not unchanged:
            raise AssertionError("Read-only preflight detected modified source artifacts")
    print("PREFLIGHT_PASSED", len(result["branch_epochs"]), len(result["whole_model_forward_checks"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify(args.current, args.reference, args.output or args.current/"replay_audit.json")


if __name__ == "__main__":
    main()
