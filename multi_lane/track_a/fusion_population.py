"""TRAIN-only population contracts for paired, same-trajectory fusion learning."""
import copy
import hashlib

import numpy as np

from multi_lane.track_a import runner

def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


METHODS = ("current_positive", "seen_population")
SALT = "emotic-reliability-calibration-v1"


def image_group(sample_id):
    value = str(sample_id)
    if not value.startswith("train:"):
        raise ValueError("Fusion populations accept official TRAIN only")
    return value.rsplit("#person=", 1)[0]


def in_heldout_group(sample_id, fraction=.2):
    group = image_group(sample_id)
    digest = hashlib.sha256(f"{SALT}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < int(fraction * (1 << 64))


def population_indices(source, task):
    if not isinstance(task, int) or not 0 <= task < len(runner.TASK_SIZES):
        raise ValueError("Invalid incremental task")
    if len(source.targets) != len(source.sample_ids) or len(set(source.sample_ids)) != len(source.sample_ids):
        raise ValueError("Misaligned or duplicate TRAIN sample identifiers")
    for name in source.sample_ids:
        image_group(name)
    current = set(runner.task_indices(task))
    seen = set(runner.seen_indices(task))
    _, positive = runner.fit_calibration_indices(source, tuple(current), .2)
    negative = [i for i, target in enumerate(source.targets)
                if set(target) & seen and not set(target) & current
                and in_heldout_group(source.sample_ids[i])]
    if set(positive) & set(negative):
        raise AssertionError("Population overlap")
    return {"current_positive": positive, "added_negative": negative,
            "seen_population": positive + negative}


def population_manifest(source, task, prior_returned_groups=()):
    selected = population_indices(source, task)
    current = set(runner.task_indices(task))
    seen = set(runner.seen_indices(task))
    prior = set(prior_returned_groups)
    groups = lambda ix: {image_group(source.sample_ids[i]) for i in ix}
    current_groups = groups(selected["current_positive"])
    added = selected["added_negative"]
    current_eligible = [i for i, y in enumerate(source.targets) if set(y) & current]
    future_only = [i for i, y in enumerate(source.targets) if set(y) and not set(y) & seen]
    unlabelled = sum(not set(y) for y in source.targets)
    return {"task": task, "fit_split": "train", "hash_salt": SALT, "fraction": .2,
            "current_classes": sorted(current), "seen_classes": sorted(seen),
            "ids": {k: [str(source.sample_ids[i]) for i in v] for k, v in selected.items()},
            "counts": {k: len(v) for k, v in selected.items()},
            "image_groups": {k: len(groups(v)) for k, v in selected.items()},
            "added_same_current_images": sum(image_group(source.sample_ids[i]) in current_groups for i in added),
            "added_other_historical_images": sum(image_group(source.sample_ids[i]) not in current_groups for i in added),
            "current_eligible_people": len(current_eligible), "future_only_excluded_people": len(future_only),
            "unlabelled_excluded_people": unlabelled,
            "historical_train_access": bool(added), "added_historical_people": len(added),
            "added_historical_image_groups": len(groups(added)),
            "future_only_H_excluded_people": sum(in_heldout_group(source.sample_ids[i]) for i in future_only),
            "prior_returned_group_exposure": {k: sum(image_group(source.sample_ids[i]) in prior for i in v)
                                             for k, v in selected.items()},
            "history_access": "added negatives have at least one previously introduced class; historical TRAIN access is required",
            "branch_replay_updates": 0, "future_only_access_for_fitting": 0,
            "current_target_all_negative": {"current_positive": 0, "seen_population": len(added)},
            "current_heldout_unseen_in_current_task": True,
            "whole_trajectory_unseen_claim": task == 0,
            "observation_cost": {"added_person_forwards_per_branch": len(added),
                                 "added_branch_forwards_total": 3*len(added),
                                 "added_float32_logit_payload_bytes": len(added)*3*len(current)*4,
                                 "raw_image_copy_required": False,
                                 "note": "Existing TRAIN images are reread; npz/ID/label/mask bytes reported after export"}}


class DualPopulationFusion:
    """Two independent detached readouts learned at the same branch barrier."""
    def __init__(self, task_sizes, device="cpu", steps=100, *, expanded_only=False):
        # Lazy import avoids a module cycle and preserves the historical fitter.
        from scripts.emotic.run_r1_fusion_return_train import ReturnFusion
        methods = ("seen_population",) if expanded_only else METHODS
        self.banks = {m: ReturnFusion(task_sizes, device, steps) for m in methods}
        self.steps, self.device, self.task = steps, device, -1

    @property
    def updates(self):
        return {m: b.updates for m, b in self.banks.items()}

    @property
    def reports(self):
        return {m: b.reports for m, b in self.banks.items()}

    def begin_task(self, task):
        if task != self.task + 1:
            raise ValueError("Consecutive tasks required")
        for bank in self.banks.values():
            bank.begin_task(task)
        self.task = task

    def fit(self, observations):
        if self.task < 0 or any(bank.updates[self.task] for bank in self.banks.values()):
            raise ValueError("Fit exactly once after beginning a task")
        if set(observations) != set(METHODS):
            raise ValueError("Both paired populations required")
        positive, seen = (observations[m] for m in METHODS)
        n = len(positive[0])
        if len(seen[0]) < n:
            raise ValueError("Seen population must include the original heldout people")
        for left, right in zip(positive, seen):
            if not np.array_equal(as_numpy(left), as_numpy(right)[:n]):
                raise ValueError("Original heldout observations must be reused exactly in both fits")
        if not np.all(as_numpy(positive[2]).sum(axis=1) > 0):
            raise ValueError("Current-positive population contains a whole-task negative")
        if not np.all(as_numpy(seen[2])[n:] == 0):
            raise ValueError("Only whole-current-task negatives may be added")
        # Validate BOTH complete inputs before either bank mutates.
        for ids, z, y, mask in observations.values():
            ids = list(ids); z, y, mask = as_numpy(z), as_numpy(y), as_numpy(mask)
            classes = self.banks["seen_population"].bank.task_sizes[self.task]
            if (not ids or len(set(ids)) != len(ids) or any(not str(i).startswith("train:") for i in ids)
                    or z.shape != (len(ids), 3, classes) or y.shape != (len(ids), classes)
                    or mask.shape != (len(ids),) or mask.dtype != np.bool_):
                raise ValueError("Invalid aligned TRAIN population")
            if not np.isfinite(z).all() or not np.isfinite(y).all():
                raise FloatingPointError("Nonfinite fusion input")
            if not np.isin(y, [0, 1]).all():
                raise ValueError("Expected binary current-task labels")
        return {m: bank.fit(*observations[m]) for m, bank in self.banks.items()}

    def snapshot(self):
        return {"format": "dual_population_fusion_v1", "task": self.task, "steps": self.steps,
                "fit_split": "train", "banks": {m: b.snapshot() for m, b in self.banks.items()}}

    def load_snapshot(self, state):
        if (self.task != -1 or state["format"] != "dual_population_fusion_v1"
                or state["fit_split"] != "train" or state["steps"] != self.steps
                or set(state["banks"]) != set(self.banks)):
            raise ValueError("Fresh matching paired fusion required for restore")
        for method in self.banks:
            self.banks[method].load_snapshot(copy.deepcopy(state["banks"][method]))
        if any(b.task != state["task"] for b in self.banks.values()):
            raise ValueError("Paired bank task mismatch")
        self.task = state["task"]
