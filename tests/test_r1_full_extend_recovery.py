"""Safety and trajectory contracts for explicit numerical recovery."""
import copy
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.emotic import run_r1_full_extend_recovery as recovery


def tiny_whole(task=0, epoch=1):
    branches = {}
    for index, view in enumerate(recovery.VIEWS):
        branches[view] = {
            "task": task,
            "config": {"seed": 1, "lr": .0125, "git": {"commit": "old"}},
            "model": {"task_selector": torch.tensor([float(index), .25])},
            "training": {
                "next_epoch": epoch,
                "history": [{"optimizer_steps": 2, "skipped_optimizer_steps": 0}
                            for _ in range(epoch)],
                "completed_task_updates": 2 * epoch,
                "optimizer": {"state": {0: {"step": torch.tensor(float(2 * epoch)),
                                              "exp_avg": torch.tensor([.125, -.5]),
                                              "exp_avg_sq": torch.tensor([.2, .3])}},
                              "param_groups": [{"params": [0], "lr": .0125}]},
                "optimizer_parameter_names": [["task_selector"]],
                "scheduler": {"last_epoch": epoch, "T_max": 30},
                "scaler": {"scale": 4096., "_growth_tracker": 728,
                           "growth_factor": 2., "backoff_factor": .5,
                           "growth_interval": 1000000000},
            },
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state(), "cuda": []},
        }
    return {"condition": "full_extend", "task": task, "epoch": epoch,
            "branches": branches, "fusion": {"bank": {}}}


class RecoveryContracts(unittest.TestCase):
    def assertNestedEqual(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertNestedEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertNestedEqual(a, b)
        else:
            self.assertEqual(left, right)

    def test_whitelist_excludes_failed_epoch_and_later_tasks(self):
        source = Path("source_trajectory")
        paths = recovery.prefix_paths(source, 3, 13)
        self.assertEqual(len(paths), len(set(paths)))
        relative = {str(p.relative_to(source)) for p in paths}
        self.assertIn("recovery/task3/epoch13.pth", relative)
        self.assertIn("recovery/task2/epoch35.pth", relative)
        self.assertIn("checkpoints/task2.pth", relative)
        self.assertNotIn("checkpoints/task3.pth", relative)
        self.assertFalse(any("task3/epoch14" in p or "task4/" in p for p in relative))
        self.assertNotIn("abort.json", relative)
        self.assertNotIn("checkpoints/latest.json", relative)
        for view, branch in recovery.branch_paths(source).items():
            self.assertIn(str((branch/"epoch_recovery/task3/epoch13.pth").relative_to(source)), relative)
            self.assertNotIn(str((branch/"epoch_recovery/task3/epoch14.pth").relative_to(source)), relative)

    def test_boundary_requires_same_task_epoch_complete_history_and_zero_skips(self):
        recovery.validate_boundary(tiny_whole(3, 13), 3, 13)
        for problem in ("condition", "task", "epoch", "branch_task", "branch_epoch", "history", "skip"):
            whole = tiny_whole(3, 13)
            state = whole["branches"]["person"]
            if problem == "condition": whole["condition"] = "holdout_return"
            elif problem == "task": whole["task"] = 4
            elif problem == "epoch": whole["epoch"] = 14
            elif problem == "branch_task": state["task"] = 4
            elif problem == "branch_epoch": state["training"]["next_epoch"] = 14
            elif problem == "history": state["training"]["history"].pop()
            else: state["training"]["history"][-1]["skipped_optimizer_steps"] = 7
            with self.assertRaises(ValueError):
                recovery.validate_boundary(whole, 3, 13)

    def test_missing_rng_or_training_recovery_fields_are_rejected(self):
        for key in ("python", "numpy", "cpu", "cuda"):
            whole = tiny_whole()
            del whole["branches"]["person"]["rng"][key]
            with self.assertRaises(ValueError):
                recovery.validate_boundary(whole, 0, 1)
        for key in ("optimizer", "scheduler", "scaler", "completed_task_updates", "optimizer_parameter_names"):
            whole = tiny_whole()
            del whole["branches"]["face"]["training"][key]
            with self.assertRaises(ValueError):
                recovery.validate_boundary(whole, 0, 1)

    def test_nonfinite_model_or_adam_cannot_be_used_as_boundary(self):
        for where in ("model", "adam"):
            whole = tiny_whole()
            state = whole["branches"]["person"]
            value = state["model"]["task_selector"] if where == "model" else state["training"]["optimizer"]["state"][0]["exp_avg"]
            value[0] = float("nan")
            with self.assertRaises(FloatingPointError):
                recovery.validate_boundary(whole, 0, 1)

    def test_smoke_prepare_copies_safe_state_exactly_and_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory)/"source", Path(directory)/"recovery"
            whole = tiny_whole()
            prefix = recovery.prefix_paths(source, 0, 1, smoke=True)
            for path in prefix:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path == source/"recovery/task0/epoch01.pth":
                    torch.save(whole, path)
                elif path.suffix == ".pth":
                    view = next(v for v, branch in recovery.branch_paths(source).items()
                                if str(path).startswith(str(branch)))
                    torch.save(whole["branches"][view], path)
                else:
                    path.write_text('{"accepted": true}\n')
            failed = source/"recovery/task0/epoch02.pth"
            failed.write_bytes(b"failed epoch must never be copied")
            before = {str(p.relative_to(source)): recovery.sha(p) for p in source.rglob("*") if p.is_file()}
            with patch.object(recovery.runner, "git_metadata", return_value={"head": "test"}):
                manifest = recovery.prepare(source, root, None, 32., smoke=True)
            self.assertTrue(manifest["same_training_trajectory"])
            self.assertTrue(manifest["optimizer_moments_and_rng_retained"])
            self.assertTrue(manifest["numerical_recovery_revalidation_not_unmodified_original_run"])
            copied = torch.load(root/"recovery/task0/epoch01.pth", map_location="cpu")
            self.assertNestedEqual(copied, whole)
            for view, branch in recovery.branch_paths(root).items():
                state = torch.load(branch/"epoch_recovery/task0/epoch01.pth", map_location="cpu")
                self.assertNestedEqual(state, whole["branches"][view])
            after = {str(p.relative_to(source)): recovery.sha(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((root/"recovery/task0/epoch02.pth").exists())
            self.assertEqual(recovery.read(root/"checkpoints/latest.json")["epoch"], 1)
            recovery.verify_prefix(root)
            with self.assertRaises(FileExistsError):
                recovery.prepare(source, root, None, 32., smoke=True)
            with self.assertRaises(FileExistsError):
                recovery.prepare(source, source, None, 32., smoke=True)
            # Integrity verification covers the source as well as its copy.
            prefix[0].write_bytes(b"unexpected source modification")
            with self.assertRaises(AssertionError):
                recovery.verify_prefix(root)

    def test_formal_recovery_requires_completed_matching_finite_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source"
            boundary = source/"recovery/task3/epoch13.pth"
            boundary.parent.mkdir(parents=True)
            torch.save(tiny_whole(3, 13), boundary)
            digest = recovery.sha(boundary)
            diagnostic = Path(directory)/"scan.json"
            valid = {"status": "complete", "all_expected_batches_scanned": True,
                     "source_sha256": digest, "view": "person",
                     "mode_nonfinite_batches": {"amp32": 0, "amp4096": 1},
                     "source_sha_unchanged": True, "model_unchanged": True, "optimizer_unchanged": True}
            for problem in ("missing", "status", "all_expected_batches_scanned", "source_sha256",
                            "view", "nonfinite", "source_sha_unchanged", "model_unchanged", "optimizer_unchanged"):
                row = copy.deepcopy(valid)
                if problem == "status": row[problem] = "partial"
                elif problem == "source_sha256": row[problem] = "wrong"
                elif problem == "view": row[problem] = "face"
                elif problem == "nonfinite": row["mode_nonfinite_batches"]["amp32"] = 1
                elif problem != "missing": row[problem] = False
                recovery.write(diagnostic, row)
                root = Path(directory)/f"new_{problem}"
                with patch.object(recovery, "BOUNDARY_SHA", digest), self.assertRaises(ValueError):
                    recovery.prepare(source, root, None if problem == "missing" else diagnostic, 32.)
                self.assertFalse(root.exists())

    def test_only_person_numerical_policy_and_git_metadata_may_change(self):
        saved = {"git": {"head": "old"}, "lr": .0125, "seed": 1, "batch_size": 64,
                 "r1_fixed_loss_scale": 4096., "r1_loss_scale_growth_interval": 1000000000}
        for view in recovery.VIEWS:
            changed = dict(saved, git={"head": "new"})
            self.assertEqual(recovery.permitted_config_delta(saved, changed, view), ["git"])
        numeric = dict(saved, r1_fixed_loss_scale=32.)
        self.assertEqual(recovery.permitted_config_delta(saved, numeric, "person"), ["r1_fixed_loss_scale"])
        for view in ("full", "face"):
            with self.assertRaises(ValueError):
                recovery.permitted_config_delta(saved, numeric, view)
        for key, value in (("lr", .001), ("seed", 2), ("batch_size", 128), ("new_recipe_key", None)):
            with self.assertRaises(ValueError):
                recovery.permitted_config_delta(saved, dict(saved, **{key: value}), "person")
        with self.assertRaises(ValueError):
            recovery.permitted_config_delta(dict(saved, optional_recipe=None), saved, "person")

    def test_runtime_preserves_original_resume_and_changes_only_person_scale(self):
        original = tiny_whole()["branches"]["person"]["training"]
        for view in recovery.VIEWS:
            resume = copy.deepcopy(original)
            model = SimpleNamespace(r1_view=view, r1_resume=resume, r1_fixed_loss_scale=4096.)
            before = copy.deepcopy(resume)
            seen = []
            def train(model, *args, **kwargs):
                self.assertIsNot(model.r1_resume, resume)
                self.assertIsNone(model.r1_fixed_loss_scale)
                expected = copy.deepcopy(before)
                if view == "person": expected["scaler"]["scale"] = 32.
                self.assertNestedEqual(expected, model.r1_resume)
                seen.append(True)
                return "trained"
            with patch.object(recovery, "read", return_value={"person_loss_scale_after_boundary": 32.}), \
                 patch.object(recovery.runner, "train_task", train), \
                 patch.object(recovery.base, "branch_flags", return_value=[]), \
                 patch.object(recovery.R1Receiver, "load_own", lambda self, saved: None):
                recovery.configure_runtime(Path("new_recovery"))
                self.assertEqual(recovery.runner.train_task(model), "trained")
            self.assertTrue(seen)
            self.assertIs(model.r1_resume, resume)
            self.assertEqual(model.r1_fixed_loss_scale, 4096.)
            self.assertNestedEqual(before, resume)


if __name__ == "__main__":
    unittest.main()
