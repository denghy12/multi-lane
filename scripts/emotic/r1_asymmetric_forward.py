"""Run the three receivers together and verify their archived outputs."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.evaluation_scores import load_evaluation_scores
from multi_lane.track_a.fuse_fixed_three_view_test import (
    LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS,
)
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import (
    model_from_config, read, sha, single_thread_worker, write,
)
from multi_lane.track_a.runner import (
    TaskMetrics, compute_metrics, dataset_view, seen_indices, set_seed, summarize_tasks,
    write_evaluation_scores,
)
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, restore, score_suffix


SOURCE = Path("../multi-lane-main-r1-patch-seed-search-20261107/output/r1_patch_seed_search/20261107")
SITES = (0, 2, 7)
GROUPS = {
    "asymmetric": ("person10", "face9"),
    "layer9": ("person9", "face9"),
    "layer10": ("person10", "face10"),
}
STATES = {
    "both_on": (True, True),
    "person_off": (False, True),
    "face_off": (True, False),
    "both_off": (False, False),
}


def branch_paths(seed):
    base = SOURCE / f"seed{seed}"
    paths = {"full": base / "native/full/train"}
    paths.update({
        f"{view}{layer}": base / f"layer{layer}/attention/{view}/train"
        for view in ("person", "face") for layer in (9, 10)
    })
    return paths


def fuse_logits(full, person, face, reliable):
    # Match the float32 CPU arithmetic used by the locked R1 readout.
    weights = np.where(
        reliable.cpu().numpy()[:, None],
        LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS,
    )
    values = (
        weights[:, 0, None] * full.numpy()
        + weights[:, 1, None] * person.numpy()
        + weights[:, 2, None] * face.numpy()
    )
    return torch.from_numpy(values)


class AsymmetricReceivers(nn.Module):
    """One batch-level inference graph with independently trained task paths."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleDict(models)

    @staticmethod
    def receiver_inputs(images, name):
        if name == "full":
            return images["evidence_full"]
        if name.startswith("person"):
            return images
        return {
            "full": images["evidence_face"],
            "evidence_full": images["evidence_full"],
            "evidence_person": images["full"],
            "evidence_face_reliable": images["evidence_face_reliable"],
        }

    def forward(self, images, attribution=False):
        names = tuple(self.models) if attribution else ("full", "person10", "face9")
        streams = {}
        for name in names:
            model = self.models[name]
            enabled_states = (True,) if name == "full" or not attribution else (True, False)
            for enabled in enabled_states:
                model.r1_enabled = enabled
                # Each call gets its own autocast cache, including ON/OFF calls.
                with torch.autocast("cuda"):
                    row = model.seen_logits(self.receiver_inputs(images, name)).float().cpu()
                if not torch.isfinite(row).all():
                    raise FloatingPointError(f"Nonfinite {name} output")
                streams[(name, enabled)] = row
            model.r1_enabled = True
        groups = GROUPS if attribution else {"asymmetric": GROUPS["asymmetric"]}
        states = STATES if attribution else {"both_on": STATES["both_on"]}
        fused = {}
        for group, (person, face) in groups.items():
            for state, (person_on, face_on) in states.items():
                fused[(group, state)] = fuse_logits(
                    streams[("full", True)], streams[(person, person_on)],
                    streams[(face, face_on)], images["evidence_face_reliable"],
                )
        return streams, fused


def verify_replay(path, ids, targets, logits):
    saved = load_evaluation_scores(path, 64)
    positions = {value: index for index, value in enumerate(saved.sample_ids)}
    order = np.asarray([positions[value] for value in ids])
    if not np.array_equal(saved.targets[order], targets.numpy()):
        raise AssertionError(f"Replay labels differ for {path}")
    difference = float(np.abs(saved.logits[order] - logits.numpy()).max())
    if difference != 0:
        raise AssertionError(f"Source replay differs by {difference}: {path}")
    return {"path": str(path), "max_abs": difference, "samples": len(ids)}


def dump_scores(path, task, ids, rows, targets):
    logits = torch.cat(rows)
    write_evaluation_scores(
        path, task, ids, logits, targets,
        probabilities=torch.cat([row.sigmoid() for row in rows]),
        batch_lengths=[len(row) for row in rows],
    )
    return logits


def evaluate(seed, output, smoke=False):
    inputs = input_paths()
    paths = branch_paths(seed)
    set_seed(seed, True)
    torch.set_num_threads(2)
    models = {
        name: model_from_config(read(path / "config.json"), inputs.clip_checkpoint)
        for name, path in paths.items()
    }
    module = AsymmetricReceivers(models).requires_grad_(False).eval()
    tasks = (0, 7) if smoke else tuple(range(8))
    hashes = {
        str(path / "compact_checkpoints" / f"task{task}.pth"):
            sha(path / "compact_checkpoints" / f"task{task}.pth")
        for path in paths.values() for task in tasks
    }
    replay = []
    records = {}
    torch.cuda.reset_peak_memory_stats()
    for split in ("val", "test"):
        source = load_source(paths["person10"], inputs, split)
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
            stream_rows, fused_rows = {}, {}
            ids, labels, reliability = [], [], []
            with torch.no_grad():
                for images, target, batch_ids in loader:
                    images = move_model_inputs(images, torch.device("cuda"))
                    streams, fused = module(images, attribution=task in SITES)
                    for key, row in streams.items():
                        stream_rows.setdefault(key, []).append(row)
                    for key, row in fused.items():
                        fused_rows.setdefault(key, []).append(row)
                    ids.extend(str(value) for value in batch_ids)
                    labels.append(target.float().cpu())
                    reliability.append(images["evidence_face_reliable"].bool().cpu())
            targets, reliable = torch.cat(labels), torch.cat(reliability)
            predictions = {}
            for (name, enabled), rows in stream_rows.items():
                mode = "on" if enabled else "off"
                destination = output / "branches" / name / mode / score_suffix(split, task)
                logits = dump_scores(destination, task, ids, rows, targets)
                predictions[(name, enabled)] = logits
                if enabled:
                    replay.append(verify_replay(
                        paths[name].parent / score_suffix(split, task), ids, targets, logits,
                    ))
            for (group, state), rows in fused_rows.items():
                destination = output / "fusion_scores" / group / state / score_suffix(split, task)
                logits = dump_scores(destination, task, ids, rows, targets)
                populations = {}
                for population, mask in (
                    ("all", torch.ones(len(ids), dtype=torch.bool)),
                    ("reliable", reliable), ("unreliable", ~reliable),
                ):
                    metric = compute_metrics(
                        task, logits[mask].sigmoid(), targets[mask], .5,
                    )
                    populations[population] = asdict(metric)
                records.setdefault(split, {}).setdefault(group, {}).setdefault(
                    state, [],
                ).append(populations)
            # Face switches must leave the fallback population bitwise unchanged.
            if task in SITES:
                for group, (person, face) in GROUPS.items():
                    for person_on in (True, False):
                        a = fuse_logits(
                            predictions[("full", True)], predictions[(person, person_on)],
                            predictions[(face, True)], reliable,
                        )
                        b = fuse_logits(
                            predictions[("full", True)], predictions[(person, person_on)],
                            predictions[(face, False)], reliable,
                        )
                        if not torch.equal(a[~reliable], b[~reliable]):
                            raise AssertionError("Face OFF changed fallback predictions")
            write(output / "progress.json", {
                "last_split": split, "last_task": task, "replayed_paths": len(replay),
            })
            print("ASYMMETRIC_FORWARD", seed, split, task, len(ids), flush=True)
    summaries = {}
    for split, groups in records.items():
        summaries[split] = {}
        for group, states in groups.items():
            summaries[split][group] = {}
            for state, rows in states.items():
                summaries[split][group][state] = {
                    pop: {
                        "metrics": [row[pop] for row in rows],
                        "summary": summarize_tasks([
                            TaskMetrics(**row[pop]) for row in rows
                        ]) if len(rows) == 8 else None,
                    }
                    for pop in ("all", "reliable", "unreliable")
                }
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError("Source checkpoint changed")
    write(output / "result.json", {
        "seed": seed, "smoke": smoke, "results": summaries,
        "source_replay": replay, "checkpoint_hashes": hashes,
    })
    write(output / "completion.json", {
        "status": "complete", "seed": seed, "smoke": smoke,
        "replayed_paths": len(replay), "all_source_logits_exact": True,
        "smoke_train_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.seed, args.output, args.smoke)
