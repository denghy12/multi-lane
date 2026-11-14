"""Calibrate a two-share readout on original-image TRAIN folds, then score frozen predictions.

Gate inputs are epoch15->epoch30 TRAIN OOF mAP and the final branch trajectory's
Validation final/mean mAP. Both candidate branches are locked before Test is read.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.fusion_population import image_group
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import TASK_SIZES, average_precision, task_indices
from multi_lane.track_a.task_logit_fusion import ReliabilityLogitFusion
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, blend

FOLD_SALT = "r1-early-fusion-train-fold-v1"
TOLERANCE_MAP = 1e-4
VIEWS = ("full", "person", "face")
TASKS = range(len(TASK_SIZES))


def train_folds(ids):
    groups = [image_group(i) for i in ids]
    return np.array([
        int.from_bytes(hashlib.sha256(f"{FOLD_SALT}:{g}".encode()).digest()[:8], "big") % 3
        for g in groups], dtype=np.int64)


def current_ap(z, y):
    scores = torch.from_numpy(z).sigmoid().numpy()
    ap = [100.0 * average_precision(scores[:, c], y[:, c]) for c in range(y.shape[1])]
    return {"per_class_ap": ap, "mAP": float(np.mean(ap))}


def class_pairs(y):
    pairs = []
    for col in y.T:
        pos = torch.where(col == 1)[0]
        neg = torch.where(col == 0)[0]
        if not len(pos) or not len(neg):
            raise ValueError("A TRAIN fitting fold lacks a positive or negative for a current class")
        pairs.append((pos, neg))
    return pairs


def pair_loss(logits, pairs):
    losses = []
    for c, (pos, neg) in enumerate(pairs):
        # All pairs are feasible for this fixed H (largest task has 3093 people).
        diff = logits[neg, c][None, :] - logits[pos, c][:, None]
        losses.append(F.softplus(diff).mean())
    return torch.stack(losses).mean()


def fit_matrix(z, y, reliable, objective, device):
    """One task, two parameters, equal view shares, 100 full-cohort Adam steps."""
    if objective not in ("bce", "pairwise"):
        raise ValueError(objective)
    if (z.shape != (len(y), 3, y.shape[1]) or reliable.shape != (len(y),)
            or reliable.dtype != np.bool_ or not np.isfinite(z).all()
            or not np.all((y == 0) | (y == 1))):
        raise ValueError("Inconsistent aligned TRAIN observations")
    logits = torch.as_tensor(z, device=device, dtype=torch.float32)
    targets = torch.as_tensor(y, device=device, dtype=torch.float32)
    mask = torch.as_tensor(reliable, device=device)
    pairs = class_pairs(targets) if objective == "pairwise" else None
    gate = ReliabilityLogitFusion().to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=.02, weight_decay=0.)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        pred = gate(logits, mask)
        loss = (F.binary_cross_entropy_with_logits(pred, targets) if objective == "bce"
                else pair_loss(pred, pairs))
        loss.backward()
        if not bool(torch.isfinite(loss.detach()) & torch.isfinite(gate.raw_shares.grad).all()):
            raise FloatingPointError("Nonfinite two-share update")
        opt.step()
    w = gate.weight_matrix().detach().cpu().numpy()
    if not np.isfinite(w).all():
        raise FloatingPointError("Nonfinite trained shares")
    return w


def read_observations(folder, task):
    views = []
    for view in VIEWS:
        with np.load(folder/f"task{task}"/f"{view}.npz", allow_pickle=False) as f:
            row = {key: f[key] for key in ("ids", "logits", "targets", "reliable")}
        if views:
            for k in ("ids", "targets", "reliable"):
                if not np.array_equal(row[k], views[0][k]):
                    raise ValueError(f"Misaligned {view} {k} in task{task}")
        views.append(row)
    base = views[0]
    if len(base["ids"]) != len(set(base["ids"])) or len(base["targets"].shape) != 2:
        raise ValueError("Invalid observation population")
    if base["targets"].shape != (len(base["ids"]), TASK_SIZES[task]):
        raise ValueError("Fit only current task classes")
    return (base["ids"], base["targets"], base["reliable"],
            np.stack([r["logits"] for r in views], axis=1))


def datasets(source, root):
    result = []
    for task in TASKS:
        audit = read(root/f"export_audit/task{task}.json")
        if (audit["status"] != "complete" or audit["task"] != task or audit["epoch"] != 15
                or not audit["epoch30_replay_exact"] or not audit["source_checkpoint_sha_unchanged"]
                or Path(audit["source_root"]).resolve() != source.resolve()):
            raise ValueError(f"Unverified epoch15 observations at task{task}")
        for view in VIEWS:
            if sha(root/f"observations/task{task}/{view}.npz") != audit["output_hashes"][view]:
                raise ValueError(f"Modified epoch15 observations at task{task}/{view}")
        ids, y, mask, z15 = read_observations(root/"observations", task)
        ids30, y30, mask30, z30 = read_observations(source/"population_observations", task)
        if any(not np.array_equal(a, b) for a, b in ((ids, ids30), (y, y30), (mask, mask30))):
            raise ValueError(f"Different epoch15/30 TRAIN cohorts in task{task}")
        manifest = read(source/f"population_manifests/full/task{task}.json")
        if list(ids) != manifest["ids"]["seen_population"]:
            raise ValueError(f"Wrong TRAIN cohort at task{task}")
        folds = train_folds(ids)
        result.append((ids, y, mask, folds, {15: z15, 30: z30}))
    return result


def out_of_fold(data, fit_epoch, objective, device):
    task_reports, full_weights = [], []
    for task, (ids, y, reliable, folds, by_epoch) in enumerate(data):
        fit_z = by_epoch[fit_epoch]
        epochs = sorted({fit_epoch, 30})
        predictions = {e: np.empty_like(y, dtype=np.float32) for e in epochs}
        control = {e: blend(by_epoch[e], reliable, FIXED) for e in epochs}
        folds_report = []
        for f in range(3):
            test = folds == f
            train = ~test
            if not train.any() or not test.any():
                raise ValueError(f"Empty TRAIN fold at {task}/{f}")
            w = fit_matrix(fit_z[train], y[train], reliable[train], objective, device)
            row = {"fold": f, "fit_people": int(train.sum()), "heldout_people": int(test.sum()),
                   "heldout_images": len({image_group(i) for i in ids[test]}),
                   "fit_positive_per_class": y[train].sum(0).astype(int).tolist(),
                   "heldout_positive_per_class": y[test].sum(0).astype(int).tolist(),
                   "heldout_reliable": int(reliable[test].sum()), "weights": w.tolist(), "readout": {}}
            for e in epochs:
                predictions[e][test] = blend(by_epoch[e][test], reliable[test], w)
                row["readout"][e] = {"learned": current_ap(predictions[e][test], y[test]),
                                     "fixed": current_ap(control[e][test], y[test])}
            folds_report.append(row)
        pooled = {e: {"learned": current_ap(predictions[e], y), "fixed": current_ap(control[e], y)}
                  for e in epochs}
        task_reports.append({"task": task, "folds": folds_report, "pooled_oof": pooled})
        full_weights.append(fit_matrix(fit_z, y, reliable, objective, device))
    return task_reports, full_weights


def stage_evaluation(source, split, banks, phase="final"):
    records = make_records(("fixed", *banks))
    anchor = {"e30_logit_abs_error": 0., "fixed_logit_abs_error": 0.}
    tasks = TASKS if phase == "final" else (0, 2, 7)
    for t in tasks:
        with np.load(source/f"predictions/{phase}/{split}/task{t}.npz") as rows:
            ids, z, y, good = [rows[k] for k in ("ids", "branch_logits", "targets", "reliable")]
            if not all(str(i).startswith(split + ":") for i in ids) or good.dtype != np.bool_:
                raise ValueError(f"Wrong {split} prediction provenance")
            fused = {"fixed": blend(z, good, FIXED)}
            for name, matrices in banks.items():
                fused[name] = np.concatenate([
                    blend(z[:, :, list(task_indices(u))], good, matrices[u])
                    for u in range(t+1)], axis=-1)
            anchor["fixed_logit_abs_error"] = max(anchor["fixed_logit_abs_error"],
                                                  float(np.max(abs(fused["fixed"]-rows["fixed"]))))
            anchor["e30_logit_abs_error"] = max(anchor["e30_logit_abs_error"],
                float(np.max(abs(fused["e30_bce"]-rows["seen_population"]))))
            record(records, t, fused, y, good, list(task_indices(t)))
    return summarize(records, phase == "pre_return"), anchor


def replay(source, split, result, anchor):
    # The result file contains both splits; no test value enters a fitting/gating expression.
    archived = read(source/"result.json")[split]
    discrepancy = max(
        abs(archived[old][pop]["metrics"][i]["mAP"]-result[new][pop]["metrics"][i]["mAP"])
        for old, new in [("fixed", "fixed"), ("seen_population", "e30_bce")]
        for pop in ("all", "reliable", "unreliable") for i in TASKS)
    anchor["max_official_mAP_point_error"] = discrepancy
    if anchor["fixed_logit_abs_error"] != 0 or anchor["e30_logit_abs_error"] > 2e-5 or discrepancy > .003:
        raise ValueError(f"e30/fixed historical replay diverged ({split}): {anchor}")
    return anchor


def gate_deltas(train_report, validation):
    transfer = np.mean([r["pooled_oof"][30]["learned"]["mAP"]-r["pooled_oof"][30]["fixed"]["mAP"]
                        for r in train_report])
    v = validation["e15_bce"]["all"]["summary"]
    fixed = validation["fixed"]["all"]["summary"]
    return {"train_crossfit_e15_to_e30_taskmean_delta": float(transfer),
            "val_e35_final_delta": v["final_mAP"]-fixed["final_mAP"],
            "val_e35_average_delta": v["average_mAP"]-fixed["average_mAP"]}


def lock_weights(path, banks):
    write(path, {k: [v.tolist() for v in matrices] for k, matrices in banks.items()})
    return sha(path)


def evaluate(source, root, device):
    out = root/"analysis"
    if (out/"result.json").exists():
        raise FileExistsError("Completed analysis already exists")
    torch.set_num_threads(2)
    early = {}
    banks = {}
    cohort = datasets(source, root)
    for epoch in (15, 30):
        method = f"e{epoch}_bce"
        early[method], banks[method] = out_of_fold(cohort, epoch, "bce", device)
    weight_path = out/"early_weights_lock.json"
    early_hash = lock_weights(weight_path, banks)
    recorded = read(source/"fusion_history.json")["seen_population"]
    max_weight_error = max(float(np.max(abs(np.array(recorded[str(t)]["weights"]) - banks["e30_bce"][t])))
                           for t in TASKS)
    if max_weight_error > 2e-6:
        raise ValueError(f"e30 BCE differs from archived shares by {max_weight_error}")
    val, anchor = stage_evaluation(source, "val", banks)
    anchor_val = replay(source, "val", val, anchor)
    deltas = gate_deltas(early["e15_bce"], val)
    usable = all(d >= -TOLERANCE_MAP for d in deltas.values())
    decision = {"early_usable": usable, **deltas, "tolerance_mAP_points": TOLERANCE_MAP,
                "early_weights_sha256": early_hash, "test_used_for_decision": False,
                "epoch15_fully_fitted_weights_used_for_e35": True,
                "oof_rows_used_for_gate": "e15-trained folds scored on epoch30 TRAIN H; mean over 8 tasks",
                "e30_max_weight_abs_error": max_weight_error,
                "next": "train_15_plus_15" if usable else "compare_epoch30_bce_vs_pairwise"}
    write(out/"decision.json", decision)
    if not usable:
        early["e30_pairwise"], banks["e30_pairwise"] = out_of_fold(cohort, 30, "pairwise", device)
        val, anchor = stage_evaluation(source, "val", banks)
        anchor_val = replay(source, "val", val, anchor)
    locked_hash = lock_weights(out/"weights_lock.json", banks)
    write(out/"oof_train.json", {"salt": FOLD_SALT, "folds": 3, "results": early})
    write(out/"val.json", val)
    test, anchor_test = stage_evaluation(source, "test", banks)
    replay(source, "test", test, anchor_test)
    if sha(out/"weights_lock.json") != locked_hash or sha(weight_path) != early_hash:
        raise RuntimeError("Locked fusion shares changed during scoring")
    diagnostics = {}
    for split in ("val","test"):
        sparse, errors = stage_evaluation(source, split, banks, phase="pre_return")
        if errors["fixed_logit_abs_error"] != 0 or errors["e30_logit_abs_error"] > 2e-5:
            raise ValueError(f"Pre-return replay diverged: {split}/{errors}")
        diagnostics[split] = sparse
    write(out/"pre_return_tasks027.json", diagnostics)
    write(out/"test.json", test)
    concise = {split: {m: {key: r["all"]["summary"][key] for key in ("final_mAP","average_mAP")}
                       for m,r in record_.items()} for split, record_ in [("val", val), ("test", test)]}
    write(out/"result.json", {"status": "complete", "decision": decision, "metrics": concise,
          "replay": {"val": anchor_val, "test": anchor_test}, "weights_sha256": locked_hash,
          "train_only_fits": True, "ranking_loss": "per-class mean over all positive-negative pairs; softplus(negative-positive)",
          "optimizer": "Adam", "steps": 100, "lr": .02, "device": str(device)})
    print("EARLY_FUSION_RESULT", concise, decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    evaluate(args.source, args.root, args.device)


if __name__ == "__main__":
    main()
