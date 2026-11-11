"""Train task-private fusion only on Frozen9 TRAIN; evaluate locked Val/Test."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import TASK_SIZES, git_metadata, task_indices
from multi_lane.track_a.task_logit_fusion import TaskLogitFusion
from scripts.emotic.r1_asymmetric_forward import verify_replay
from scripts.emotic.run_r1_frozen9_fusion_ap import candidate_map
from scripts.emotic.run_r1_frozen9_fusion_backprop import rank_support, smooth_ap_backward
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen9_learned_fusion import (
    ARCHIVE, FIXED, ROOT as SOURCE_ROOT, Rows, archive_files, archived_rows,
    bce, blend, validate_rows, verify_hashes,
)
from scripts.emotic.run_r1_frozen_cross_view import input_paths, score_suffix


ROOT = Path("output/r1_train_only_fusion/seed1_20261111")
TRAIN_ROOT = Path("../multi-lane-main-r1-frozen9-learned-fusion-20261110") / SOURCE_ROOT
METHODS = ("fixed", "equal", "train_smooth_ap")
STEPS, LR, TEMPERATURE, CHUNK = 300, .02, .15, 128


def load_train(root, task):
    with np.load(root / "train_logits" / f"task{task}.npz", allow_pickle=False) as data:
        if not np.array_equal(data["class_indices"], task_indices(task)):
            raise ValueError("TRAIN cache class columns differ from this task")
        rows = Rows(data["ids"], data["logits"], data["targets"], data["reliable"])
    validate_rows(rows, task_indices(task), "train")
    return rows


def fit_task(rows, task, model, *, steps=STEPS, device="cpu"):
    """The only optimization entry point accepts TRAIN IDs and current classes."""
    validate_rows(rows, task_indices(task), "train")
    if not rows.reliable.any() or rows.reliable.all():
        raise ValueError("Both Face reliability groups are required for this experiment")
    previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gate = model.add_task(task).to(device)
    logits = torch.as_tensor(rows.logits, dtype=torch.float32, device=device).detach()
    labels = torch.as_tensor(rows.targets, dtype=torch.float32, device=device)
    reliable = torch.as_tensor(rows.reliable, device=device)
    support = rank_support(labels)
    if not support:
        raise ValueError("No current class has positive and negative TRAIN support")
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=0)
    trace = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        z = gate(logits, reliable)
        loss = smooth_ap_backward(z, support, chunk=CHUNK, temperature=TEMPERATURE)
        if not torch.isfinite(loss) or not torch.isfinite(gate.raw_shares.grad).all():
            raise FloatingPointError("Nonfinite TRAIN fusion objective or gradient")
        if step in (0, 1, 29, 99, steps):
            prediction = z.detach().cpu().numpy()
            trace.append({
                "step": step, "smooth_ap_loss": float(loss),
                "real_mAP": float(candidate_map(prediction[None], rows.targets)[0]),
                "bce": bce(prediction.astype(np.float64), rows.targets),
                "gradient_norm": float(gate.raw_shares.grad.norm()),
                "shares": gate.raw_shares.sigmoid().detach().cpu().tolist(),
            })
        if step < steps:
            optimizer.step()
    for key, value in previous.items():
        if not torch.equal(value, model.state_dict()[key]):
            raise AssertionError("An old task fusion parameter changed")
    gate.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    stats = {
        "task": task, "fit_split": "train", "people": len(rows.ids),
        "reliable": int(rows.reliable.sum()), "unreliable": int((~rows.reliable).sum()),
        "all_current_negative": int((rows.targets.sum(axis=1) == 0).sum()),
        "positives_per_class": rows.targets.sum(axis=0).astype(int).tolist(),
        "negatives_per_class": (1-rows.targets).sum(axis=0).astype(int).tolist(),
        "ranking_classes": [int(c) for c, _, _ in support],
        "trace": trace, "old_parameter_exact": True,
        "weights": gate.weight_matrix().detach().cpu().tolist(),
    }
    return stats, copy.deepcopy(optimizer.state_dict())


def save_checkpoint(path, model, optimizers, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pth")
    torch.save({
        "format": "task_logit_fusion_train_only_v1", "fit_split": "train",
        "task_sizes": model.task_sizes, "tasks_trained": len(model.tasks),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_states": optimizers, "provenance": provenance,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if state["fit_split"] != "train" or state["format"] != "task_logit_fusion_train_only_v1":
        raise ValueError("Unexpected fusion checkpoint provenance")
    model = TaskLogitFusion(state["task_sizes"])
    for task in range(state["tasks_trained"]):
        model.add_task(task)
    model.to(device).load_state_dict(state["model_state_dict"], strict=True)
    return model.seal()


@torch.no_grad()
def evaluate(split, model, manifest, tasks, output):
    if split not in ("val", "test") or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Evaluation requires a sealed TRAIN model and Val/Test split")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    device = next(model.parameters()).device
    reliable_lookup = load_face_reliability(manifest, split)
    records, replays, roundtrips = make_records(METHODS), [], []
    equal = np.asarray([[.5, .5, 0], [1/3, 1/3, 1/3]], dtype=np.float32)
    for task in tasks:
        rows = archived_rows(split, task, reliable_lookup)
        logits = torch.as_tensor(rows.logits, device=device)
        reliable = torch.as_tensor(rows.reliable, device=device)
        learned = model(logits, reliable, task).cpu().numpy()
        matrices = [model.tasks[t].weight_matrix().cpu().numpy() for t in range(task+1)]
        reconstructed = np.concatenate([
            blend(rows.logits[:, :, list(task_indices(t))], rows.reliable, matrices[t])
            for t in range(task+1)
        ], axis=1)
        if not np.array_equal(learned, reconstructed):
            raise AssertionError("Fusion module did not replay its saved weight matrix")
        fused = {"fixed": blend(rows.logits, rows.reliable, FIXED),
                 "equal": blend(rows.logits, rows.reliable, equal),
                 "train_smooth_ap": learned}
        reference = ARCHIVE / "fusion_scores/D_both_edges" / score_suffix(split, task)
        replays.append(verify_replay(reference, rows.ids, torch.from_numpy(rows.targets),
                                     torch.from_numpy(fused["fixed"])))
        record(records, task, fused, rows.targets, rows.reliable, list(task_indices(task)))
        path = output / "predictions" / split / f"task{task}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=rows.ids, targets=rows.targets, reliable=rows.reliable,
                            **fused)
        roundtrips.append({"task": task, "module_matrix_max_abs": 0.})
        print("EVALUATE", split, task, len(rows.ids), flush=True)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise AssertionError("Evaluation mutated TRAIN parameters")
    return {"results": summarize(records, len(tasks) != len(TASK_SIZES)),
            "fixed_exact_replay": replays, "module_roundtrip": roundtrips,
            "evaluation_parameter_change": 0.}


def select(validation, smoke):
    if smoke:
        return {"selected": None, "smoke": True}
    results = validation["results"]
    base = results["fixed"]["all"]["summary"]
    candidate = results["train_smooth_ap"]["all"]["summary"]
    eligible = all(candidate[k] > base[k] for k in ("final_mAP", "average_mAP"))
    return {"selected": "train_smooth_ap" if eligible else "fixed", "eligible": eligible,
            "criterion": "both complete Validation mAPs above original fixed",
            "validation_updates": False}


def verify_image_forward(model):
    # Source image logits were produced by set_seed(1, True). The fusion optimizer
    # uses FP32/TF32-off separately; keep the two numerical protocols explicit.
    from multi_lane.track_a.runner import set_seed
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        set_seed(1, True)
        return _verify_image_forward(model)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def _verify_image_forward(model):
    """Smoke: real image forward replays archived branches and learned fusion."""
    from multi_lane.track_a.r1_crossvit_experiment import model_from_config
    from scripts.emotic.r1_frozen9_edge_forward import ROOT as EDGE_ROOT, branch_paths
    from scripts.emotic.run_r1_frozen9_learned_fusion import BRANCHES, export_rows
    from scripts.emotic.run_r1_frozen_cross_view import load_source

    inputs = input_paths()
    paths = branch_paths(EDGE_ROOT, "readonly")
    models = {name: model_from_config(read(path / "config.json"), inputs.clip_checkpoint)
              for name, path in paths.items()}
    source = load_source(paths["person_all"], inputs, "val")
    report = []
    for task in (0, 1):
        rows = export_rows(models, paths, source, "val", task, smoke=True)
        for index, name in enumerate(BRANCHES):
            report.append(verify_replay(
                ARCHIVE / "branches" / name / score_suffix("val", task), rows.ids,
                torch.from_numpy(rows.targets), torch.from_numpy(rows.logits[:, index]),
            ))
        lookup = load_face_reliability(inputs.face_manifest_root, "val")
        cached = archived_rows("val", task, lookup)
        mapping = {value: i for i, value in enumerate(cached.ids)}
        indices = np.array([mapping[value] for value in rows.ids])
        device = next(model.parameters()).device
        with torch.no_grad():
            actual = model(torch.as_tensor(rows.logits, device=device),
                           torch.as_tensor(rows.reliable, device=device), task)
            expected = model(torch.as_tensor(cached.logits[indices], device=device),
                             torch.as_tensor(cached.reliable[indices], device=device), task)
        if not torch.equal(actual, expected):
            raise AssertionError("Real image learned fusion differs from cached forward")
    return {"branch_replays": report, "learned_fusion_max_abs": 0., "source_image_tf32": True,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}


def run(root, *, train_root=TRAIN_ROOT, smoke=False, device="cuda"):
    started = time.monotonic()
    if root.exists() and any(p.name != "smoke" for p in root.iterdir()):
        raise FileExistsError(f"An experiment already exists at {root}")
    tasks = tuple(range(2 if smoke else len(TASK_SIZES)))
    inputs = input_paths()
    source = read(train_root / "protocol.json")
    if source["fit_split"] != "TRAIN" or source["smoke"] or source["seed"] != 1:
        raise ValueError("Expected the audited complete seed1 TRAIN export")
    verify_hashes(source["hashes"])
    train_hashes = {str(p): sha(p) for p in [train_root / "protocol.json",
                    *[train_root / "train_logits" / f"task{t}.npz" for t in tasks]]}
    protocol = {
        "phase": "A/B: cached TRAIN-only fusion, no new branch training",
        "seed": 1, "tasks": tasks, "smoke": smoke, "fit_split": "train",
        "validation_role": "evaluation and configuration selection only; never refit",
        "test_role": "evaluation after TRAIN weights and Validation decision are locked",
        "model": "Frozen9 patch Person+Face with task-private two-parameter fusion",
        "optimizer": "FP32 Adam", "lr": LR, "steps": 10 if smoke else STEPS,
        "temperature": TEMPERATURE, "positive_chunk": CHUNK, "weight_decay": 0,
        "initial_shares": [.5, 1/3], "device": str(device), "methods": METHODS,
        "train_source_hashes": train_hashes, "original_source_hashes": source["hashes"],
        "git": git_metadata(Path.cwd()),
    }
    write(root / "protocol.json", protocol)
    model = TaskLogitFusion(TASK_SIZES)
    report, optimizers = {}, {}
    for task in tasks:
        write(root / "progress.json", {"state": "train_fusion", "task": task})
        rows = load_train(train_root, task)
        report[task], optimizers[task] = fit_task(
            rows, task, model, steps=protocol["steps"], device=device,
        )
        save_checkpoint(root / "checkpoints" / f"task{task}.pth", model, optimizers, protocol)
        write(root / "fit_report.json", report)
        print("FIT_TRAIN", task, report[task], flush=True)
    model.seal()
    checkpoint = root / "checkpoints" / f"task{tasks[-1]}.pth"
    checkpoint_hash = sha(checkpoint)
    restored = load_checkpoint(checkpoint, device)
    if any(not torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()):
        raise AssertionError("Saved fusion checkpoint differs from the trained model")
    write(root / "locked_train_weights.json", {
        "fit_split": "train", "checkpoint": str(checkpoint), "sha256": checkpoint_hash,
        "weights": {t: model.tasks[t].weight_matrix().cpu().tolist() for t in tasks},
        "fit_report_sha256": sha(root / "fit_report.json"),
    })
    lock_hash = sha(root / "locked_train_weights.json")
    validation = evaluate("val", restored, inputs.face_manifest_root, tasks, root)
    write(root / "validation.json", validation)
    decision = select(validation, smoke)
    write(root / "selection.json", decision)
    # All compared methods are locked already. No full-Val fitting operation exists.
    test_hashes = {str(p): sha(p) for p in archive_files("test", tasks)}
    test = evaluate("test", restored, inputs.face_manifest_root, tasks, root)
    write(root / "test.json", test)
    image_check = verify_image_forward(restored) if smoke else None
    verify_hashes({**source["hashes"], **train_hashes, **test_hashes,
                   str(checkpoint): checkpoint_hash,
                   str(root / "locked_train_weights.json"): lock_hash})
    write(root / "completion.json", {
        "status": "complete", "smoke": smoke, "fit_split": "train",
        "elapsed_seconds": time.monotonic() - started, "selection": decision,
        "train_parameter_count": sum(p.numel() for p in model.parameters()),
        "baseline_exact_replays": len(validation["fixed_exact_replay"]) + len(test["fixed_exact_replay"]),
        "evaluation_parameter_change": 0., "source_hashes_unchanged": True,
        "image_forward_smoke": image_check,
        "test_source_hashes": test_hashes,
        "summaries": {split: {m: groups["all"]["summary"] for m, groups in data["results"].items()}
                      for split, data in (("val", validation), ("test", test))},
    })
    write(root / "progress.json", {"state": "complete"})
    print("COMPLETE", root, round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.root / "smoke" if args.smoke else args.root,
        train_root=args.train_root, smoke=args.smoke, device=args.device)
