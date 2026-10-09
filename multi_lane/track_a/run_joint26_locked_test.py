"""Evaluate four final checkpoints now and queued stable-AMP seeds when ready."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .compare_joint26 import METHODS, TITLES, require


def compare_cohort(root):
    results, reference = {}, None
    for method in METHODS:
        result = json.loads((root / method / "seed_summary.json").read_text())
        require(result["status"] == "complete" and result["source_unchanged"]
                and result["optimizer_updates"] == 0 and not result["test_weight_search"], "Test audit failed")
        with np.load(root / method / "test_scores" / "task0.npz") as z:
            if reference is None:
                reference = (z["sample_ids"].copy(), z["targets"].copy())
            else:
                require(np.array_equal(reference[0], z["sample_ids"]) and np.array_equal(reference[1], z["targets"]), "Test cohorts mismatch")
            require(z["targets"].shape[1] == 26, "Test must cover 26 classes")
        results[method] = result
    rows = ["# 同一最终模型的 Validation / Test", "", "固定第30轮checkpoint；仅推理，没有训练、权重搜索或阈值调整。", "",
            "| 模型修改 | Validation mAP | Test mAP | Test cF1 | Test oF1 |", "| --- | ---: | ---: | ---: | ---: |"]
    rows += [f"| {TITLES[m]} | {r['source_validation_mAP']:.4f} | {r['metrics']['final_mAP']:.4f} | {r['metrics']['final_cF1']:.4f} | {r['metrics']['final_oF1']:.4f} |" for m, r in results.items()]
    (root / "comparison.md").write_text("\n".join(rows) + "\n")
    (root / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    return results


def run(args):
    require(not args.output_root.exists(), "Test batch already exists")
    args.output_root.mkdir(parents=True)
    cohorts = [("original_seed0", args.original_batch_root)] + [
        (f"stableamp_seed{seed}", args.stable_result_root / f"{args.stable_prefix}_seed{seed}") for seed in (0, 1, 2)]
    (args.output_root / "manifest.json").write_text(json.dumps(dict(
        cohorts={name: str(path) for name, path in cohorts}, gpu=args.gpu, batch_size="same_as_source_64",
        threshold="same_as_source_0.5", training_performed=False, checkpoint_selection="final_epoch_30",
        source_groups_never_mixed=True, data_root=str(args.data_root), clip_checkpoint=str(args.clip_checkpoint),
        face_manifest_root=str(args.face_manifest_root)), indent=2) + "\n")
    logs = args.log_root
    logs.mkdir(parents=True, exist_ok=True)
    for name, source_root in cohorts:
        status_path = args.output_root / "status.json"
        status_path.write_text(json.dumps(dict(status="waiting_for_final_checkpoints", cohort=name)) + "\n")
        print(f"WAITING cohort={name} source={source_root}", flush=True)
        deadline = time.monotonic() + args.wait_hours * 3600
        while not all((source_root / m / "seed_summary.json").is_file() for m in METHODS):
            require(time.monotonic() < deadline, f"Timeout waiting for {name}")
            control = source_root.parent.parent / "emotic_joint26_control" / source_root.name / "status"
            for failure in control.glob("*.exit_code"):
                require(failure.read_text().strip() == "0", f"Source training failed: {failure}")
            time.sleep(45)
        while True:
            free = int(subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                       "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            if free >= 20000:
                break
            status_path.write_text(json.dumps(dict(status="waiting_for_gpu", cohort=name, free_mib=free)) + "\n")
            require(time.monotonic() < deadline, "Timeout waiting for evaluation GPU")
            time.sleep(45)
        cohort_root = args.output_root / name
        status_path.write_text(json.dumps(dict(status="evaluating", cohort=name)) + "\n")
        processes = []
        try:
            for method in METHODS:
                handle = (logs / f"{name}_{method}.log").open("w")
                cmd = [sys.executable, "-m", "multi_lane.track_a.evaluate_joint26_checkpoint",
                       "--source", str(source_root / method), "--output", str(cohort_root / method),
                       "--data-root", str(args.data_root), "--clip-checkpoint", str(args.clip_checkpoint),
                       "--face-manifest-root", str(args.face_manifest_root)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
                processes.append((method, subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
            codes = [(method, process.wait()) for method, process, _ in processes]
            require(all(code == 0 for _, code in codes), f"Test evaluators failed: {codes}")
        finally:
            for _, _, handle in processes:
                handle.close()
        compare_cohort(cohort_root)
        print(f"TEST_COHORT_COMPLETE {name}", flush=True)
    (args.output_root / "status.json").write_text(json.dumps(dict(status="complete", cohorts=[name for name, _ in cohorts])) + "\n")
    print("JOINT26_ALL_LOCKED_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("original-batch-root", "stable-result-root", "output-root", "log-root", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stable-prefix", required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--wait-hours", type=float, default=12)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output_root.is_dir():
            (args.output_root / "status.json").write_text(json.dumps(dict(status="failed", error=str(error))) + "\n")
        raise
