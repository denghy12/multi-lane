"""Evaluate the four Full/auxiliary evidence patterns on aligned view inputs."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import (
    model_from_config, read, sha, single_thread_worker, write,
)
from multi_lane.track_a.runner import (
    TaskMetrics, compute_metrics, dataset_view, seen_indices, set_seed, summarize_tasks,
)
from scripts.emotic.r1_asymmetric_forward import (
    AsymmetricReceivers, dump_scores, fuse_logits, verify_replay,
)
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, restore, score_suffix
from scripts.emotic.run_r1_patch_receiver_depth import LAYER9, REFERENCE


ROOT = Path("output/r1_frozen9_edges/seed1_20261108")
GROUPS = {
    "A_full_only": ("person_full", "face_full"),
    "B_person_to_face": ("person_full", "face_all"),
    "C_face_to_person": ("person_all", "face_full"),
    "D_both_edges": ("person_all", "face_all"),
}


def branch_paths(root, phase):
    paths = {
        "full": REFERENCE / "native/full/train",
        "person_all": LAYER9 / "patch/attention/person/train",
        "face_all": LAYER9 / "patch/attention/face/train",
    }
    if phase == "trained":
        paths.update({
            f"{view}_full": root / "full_only/attention" / view / "train"
            for view in ("person", "face")
        })
    return paths


class EdgeReceivers(nn.Module):
    def __init__(self, models, phase):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.phase = phase

    def forward(self, images):
        outputs = {}
        for name, model in self.models.items():
            modes = ("all", "full") if self.phase == "readonly" and name != "full" else (None,)
            original = getattr(model, "frozen_cross_evidence", None)
            try:
                for mode in modes:
                    output_name = name if mode is None else name.replace("_all", "_" + mode)
                    if mode is not None:
                        model.frozen_cross_evidence = mode
                    with torch.autocast("cuda"):
                        logits = model.seen_logits(
                            AsymmetricReceivers.receiver_inputs(images, name)
                        ).float().cpu()
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"Nonfinite output from {output_name}")
                    outputs[output_name] = logits
            finally:
                if original is not None:
                    model.frozen_cross_evidence = original
        fused = {
            group: fuse_logits(
                outputs["full"], outputs[person], outputs[face],
                images["evidence_face_reliable"],
            )
            for group, (person, face) in GROUPS.items()
        }
        unreliable = ~images["evidence_face_reliable"].bool().cpu()
        for left, right in (("A_full_only", "B_person_to_face"), ("C_face_to_person", "D_both_edges")):
            if not torch.equal(fused[left][unreliable], fused[right][unreliable]):
                raise AssertionError("Face source choice changed the fallback population")
        return outputs, fused


def evaluate(root, phase, smoke=False):
    output = root / ("smoke" if smoke else "evaluation") / phase
    paths = branch_paths(root, phase)
    inputs = input_paths()
    set_seed(1, True)
    torch.set_num_threads(2)
    models = {
        name: model_from_config(read(train / "config.json"), inputs.clip_checkpoint)
        for name, train in paths.items()
    }
    module = EdgeReceivers(models, phase).requires_grad_(False).eval()
    tasks = (0, 7) if smoke else tuple(range(8))
    hashes = {
        str(train / "compact_checkpoints" / f"task{task}.pth"):
            sha(train / "compact_checkpoints" / f"task{task}.pth")
        for train in paths.values() for task in tasks
    }
    records, replay = {}, []
    torch.cuda.reset_peak_memory_stats()
    for split in ("val", "test"):
        source = load_source(paths["person_all"], inputs, split)
        for task in tasks:
            for name, model in models.items():
                restore(model, paths[name], task)
                model.requires_grad_(False).eval()
            data = dataset_view(source, seen_indices(task), include_sample_id=True)
            if smoke:
                data = Subset(data, range(min(64, len(data))))
            loader = DataLoader(
                data, batch_size=64, num_workers=2, prefetch_factor=1,
                worker_init_fn=single_thread_worker,
                generator=torch.Generator().manual_seed(77191 + task),
            )
            branch_rows, fused_rows = {}, {}
            ids, targets, masks = [], [], []
            with torch.no_grad():
                for images, target, sample_ids in loader:
                    images = move_model_inputs(images, torch.device("cuda"))
                    branches, fused = module(images)
                    for name, values in branches.items():
                        branch_rows.setdefault(name, []).append(values)
                    for name, values in fused.items():
                        fused_rows.setdefault(name, []).append(values)
                    ids.extend(str(value) for value in sample_ids)
                    targets.append(target.float().cpu())
                    masks.append(images["evidence_face_reliable"].bool().cpu())
            targets, reliable = torch.cat(targets), torch.cat(masks)
            for name, rows in branch_rows.items():
                logits = dump_scores(
                    output / "branches" / name / score_suffix(split, task),
                    task, ids, rows, targets,
                )
                if name in paths:
                    replay.append(verify_replay(
                        paths[name].parent / score_suffix(split, task),
                        ids, targets, logits,
                    ))
            for name, rows in fused_rows.items():
                logits = dump_scores(
                    output / "fusion_scores" / name / score_suffix(split, task),
                    task, ids, rows, targets,
                )
                for population, mask in (
                    ("all", torch.ones(len(ids), dtype=torch.bool)),
                    ("reliable", reliable), ("unreliable", ~reliable),
                ):
                    metric = compute_metrics(task, logits[mask].sigmoid(), targets[mask], .5)
                    records.setdefault(split, {}).setdefault(name, {}).setdefault(
                        population, [],
                    ).append(asdict(metric))
            write(output / "progress.json", {
                "split": split, "task": task, "source_replays": len(replay),
            })
            print("EDGE_FORWARD", phase, split, task, len(ids), flush=True)
    results = {
        split: {
            group: {
                pop: {
                    "metrics": rows,
                    "summary": summarize_tasks([TaskMetrics(**row) for row in rows])
                    if len(rows) == 8 else None,
                }
                for pop, rows in populations.items()
            }
            for group, populations in groups.items()
        }
        for split, groups in records.items()
    }
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError("Source checkpoints changed")
    write(output / "result.json", {
        "phase": phase, "smoke": smoke, "seed": 1, "results": results,
        "source_replays": replay, "checkpoint_hashes": hashes,
    })
    write(output / "completion.json", {
        "status": "complete", "source_replays": len(replay),
        "all_source_logits_exact": True,
        "smoke_train_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("readonly", "trained"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.root, args.phase, args.smoke)
