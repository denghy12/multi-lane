"""Verify identities and provenance before reusing the three Task10 feature caches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES


def read(path):
    return json.loads(path.read_text())


def run(prior, caran, splits):
    probe_lock_path = prior / "probe" / "validation_lock.json"
    probe_lock = read(probe_lock_path)
    caran_lock_path = caran / "caran_fit_lock.json"
    caran_lock = read(caran_lock_path)
    tasks = (0, 3, 6)
    if (probe_lock["protocol"]["tasks"] != list(tasks)
            or caran_lock["protocol"]["seed"] != 1
            or read(caran / "validation_seal.json")["caran_fit_lock_sha256"] != sha256_file(caran_lock_path)):
        raise ValueError("Expected sealed seed1 CARAN and three-task Task10 experiments")
    if "test" in splits:
        test = read(prior / "probe" / "test_results.json")
        if test["validation_lock_sha256"] != sha256_file(probe_lock_path):
            raise ValueError("Prior Test run does not refer to the same Validation lock")

    for task in tasks:
        train_record = caran_lock["tasks"][str(task)]
        for split in splits:
            rel = Path("features") / f"task{task}" / split
            path = (prior / rel).with_suffix(".npz")
            meta = read(path.with_suffix(".json"))
            peer = read((caran / rel).with_suffix(".json"))
            p = meta["protocol"]
            digest = sha256_file(path)
            locked = (test["test_features_sha256"][str(task)] if split == "test"
                      else probe_lock["train_val_hash"][str(task)][split])
            common = dict(task=task, split=split, smoke=False, batch_size=64,
                          checkpoint_sha256=train_record["checkpoint_sha256"],
                          source_config_sha256=caran_lock["source_config_sha256"])
            if (any(protocol.get(k) != v for protocol in (p, peer["protocol"]) for k, v in common.items())
                    or not meta["visibility_mask_applied"] or meta["exact_eval_replay"] != (split != "train")
                    or not peer["visibility_mask_applied"] or not peer["exact_replay"]
                    or digest != locked or digest != meta["features_sha256"]):
                raise ValueError(f"Cached features or historical protocol differ for {rel}")
            reference = digest if split == "train" else p["source_cohort_or_archive_sha256"]
            if peer["protocol"]["reference_sha256"] != reference:
                raise ValueError(f"CARAN features have a different reference for {rel}")
            if split == "train" and (p["source_cohort_or_archive_sha256"] != train_record["cohort_sha256"]
                                     or meta["people"] != train_record["people"]
                                     or meta["reliable_face"] != train_record["reliable_face"]):
                raise ValueError(f"TRAIN cohort differs from the CARAN training record for {rel}")
            with np.load(path, allow_pickle=False) as data:
                ids = data["sample_ids"]
                valid, labels = data["reliable"], data["targets"]
                n = len(ids)
                id_hash = hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest()
                if (len(set(ids)) != n or valid.shape != (n,) or valid.dtype != np.bool_
                        or labels.shape != (n, TASK_SIZES[task]) or not np.isin(labels, [0, 1]).all()
                        or data["branch_logits"].shape != (n, TASK_SIZES[task], 3)
                        or split == "train" and not labels.any(axis=1).all()):
                    raise ValueError(f"Unexpected people, targets, or current-head logit axes in {rel}")
                for record in (meta, peer):
                    if (n != record["people"] or valid.sum() != record["reliable_face"]
                            or id_hash != record["sample_id_sha256"]):
                        raise ValueError(f"People or Face availability mismatch for {rel}")
            print(f"verified {rel} people={n} reliable={int(valid.sum())} sha256={digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--caran", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    run(args.prior, args.caran, args.splits)
