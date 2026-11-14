"""Measure branch coverage and fusion sampling sensitivity using only TRAIN people."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def groups(ids):
    if not all(str(i).startswith("train:") and "#person=" in str(i) for i in ids):
        raise ValueError("Expected TRAIN person identifiers")
    return np.asarray([str(i).rsplit("#person=", 1)[0] for i in ids])


def holdout(ids, fraction):
    tokens = [hashlib.sha256(("emotic-reliability-calibration-v1:" + i).encode()).digest() for i in groups(ids)]
    cutoff = int(fraction * (1 << 64))
    return np.array([int.from_bytes(token[:8], "big") < cutoff for token in tokens])


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def census(native, source):
    """Per-task counts are evaluated using only classes introduced by that task."""
    result, seen = {}, {}
    for t in range(8):
        with np.load(native / "observations" / f"task{t}" / "epoch30_full.npz") as d:
            ids, y, valid = d["ids"], d["targets"], d["reliable"]
        if len(set(ids)) != len(ids) or not y.any(axis=1).all():
            raise ValueError(f"Incomplete current-task population {t}")
        for i, r in zip(ids, valid):
            if i in seen and seen[i] != bool(r):
                raise ValueError("Inconsistent face reliability")
            seen[i] = bool(r)
        seen_ids = np.asarray(list(seen))
        reliable = np.array(list(seen.values()))
        manifest = json.loads((source / "population_manifests/full" / f"task{t}.json").read_text())
        if set(ids[holdout(ids, .2)]) != set(manifest["ids"]["current_positive"]):
            raise AssertionError("Original current-positive split does not replay")
        if set(seen_ids[holdout(seen_ids,.2)]) != set(manifest["ids"]["seen_population"]):
            raise AssertionError("Original expanded split does not replay")
        result[t] = {}
        for a, f in [(70,.3),(80,.2),(90,.1)]:
            h, sh = holdout(ids, f), holdout(seen_ids, f)
            if set(groups(ids[h])) & set(groups(ids[~h])):
                raise AssertionError("Overlapping original images")
            result[t][a] = {
                "branch_people": int((~h).sum()), "current_H_people": int(h.sum()),
                "branch_groups": len(set(groups(ids[~h]))), "current_H_groups": len(set(groups(ids[h]))),
                "expanded_H_people": int(sh.sum()), "expanded_H_groups": len(set(groups(seen_ids[sh]))),
                "expanded_H_reliable": int((sh & reliable).sum()), "expanded_H_unreliable": int((sh & ~reliable).sum()),
                "current_H_positive_labels_by_class": y[h].sum(0).astype(int).tolist(),
                "current_H_reliable_positive_labels_by_class": y[h & valid].sum(0).astype(int).tolist(),
                "branch_positive_labels_by_class": y[~h].sum(0).astype(int).tolist()}
    return result


def sensitivity(source, device):
    from scripts.emotic.evaluate_r1_early_fusion import train_folds, fit_matrix, current_ap
    from scripts.emotic.run_r1_frozen9_learned_fusion import blend
    tasks = {}
    for t in range(8):
        rows = []
        for v in ("full", "person", "face"):
            with np.load(source / "population_observations" / f"task{t}/{v}.npz") as d:
                rows.append({k:d[k] for k in d.files})
        a = rows[0]
        if not all(np.array_equal(a[k], row[k]) for k in ("targets","reliable","ids") for row in rows):
            raise AssertionError("Unaligned branch logits")
        ids,y,r = a["ids"],a["targets"],a["reliable"]
        z = np.stack([row["logits"] for row in rows],1)
        folds = train_folds(ids)
        tasks[t] = {}
        for pct in (10,20):
            subset = holdout(ids, pct/100)
            pred = np.zeros_like(y)
            weights, counts = [], []
            for f in range(3):
                test = folds == f
                fit = subset & ~test
                if set(groups(ids[fit])) & set(groups(ids[test])):
                    raise ValueError("Train/evaluation group overlap")
                w = np.asarray(fit_matrix(z[fit],y[fit],r[fit],"bce",device),dtype=np.float32)
                weights.append(w.tolist())
                counts.append({"fit":int(fit.sum()), "evaluate":int(test.sum())})
                pred[test] = blend(z[test], r[test], w)
            tasks[t][pct] = {"oof":current_ap(pred,y), "fold_weights":weights, "fold_counts":counts,
                            "b_reliable_min_max":[float(np.min(np.asarray(weights)[:,1,2])),float(np.max(np.asarray(weights)[:,1,2]))]}
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["counts","sensitivity"])
    p.add_argument("--native", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.action == "counts" and args.native is None:
        p.error("--native supplies the archived TRAIN-only label census")
    result = census(args.native,args.source) if args.action=="counts" else sensitivity(args.source,args.device)
    save(args.out, {"action":args.action, "train_only":True, "task_results":result,
         "interpretation":"Counts are sample support, not model scores. Sensitivity uses the same 80%-trained branches and identical three held-out evaluation folds; it tests fusion sample size only."})


if __name__ == "__main__":
    main()
