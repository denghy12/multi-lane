"""FP32 continuation changes numerical execution, never the branch recipe."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.emotic import run_r1_full_extend_recovery as recovery
from tests import test_r1_full_extend_recovery as old_contracts


def probe_state(task=3, epoch=14, steps=156):
    value = old_contracts.tiny_whole(task, epoch)["branches"]["person"]
    value["config"]["amp"] = False
    value["training"]["scaler"] = {}
    value["training"]["history"][-1]["optimizer_steps"] = steps
    value["training"]["history"][-1]["skipped_optimizer_steps"] = 0
    value["training"]["history"][-1]["current_loss"] = .6
    value["training"]["history"][-1]["adapter_objective_loss"] = .02
    value["training"]["completed_task_updates"] = 2028 + steps
    return value


class FP32RecoveryContracts(unittest.TestCase):
    assertNestedEqual = old_contracts.RecoveryContracts.assertNestedEqual

    def test_fp32_flag_is_person_only_and_removes_fixed_amp_scale(self):
        original = ["--seed", "1", "--r1-fixed-loss-scale", "4096.0", "--epochs", "35"]
        manifest = {"person_precision": "fp32", "person_loss_scale_after_boundary": None}
        with patch.object(recovery, "read", return_value=manifest), \
             patch.object(recovery.base, "branch_flags", side_effect=lambda *a, **kw: list(original)), \
             patch.object(recovery.runner, "train_task", lambda *a, **kw: None), \
             patch.object(recovery.R1Receiver, "load_own", lambda *a, **kw: None):
            recovery.configure_runtime(Path("test_recovery"))
            person = recovery.base.branch_flags(Path("test_recovery"), "person")
            self.assertIn("--no-amp", person)
            self.assertNotIn("--r1-fixed-loss-scale", person)
            self.assertNotIn("4096.0", person)
            for view in ("full", "face"):
                self.assertEqual(recovery.base.branch_flags(Path("test_recovery"), view), original)
            self.assertEqual(person[person.index("--seed") + 1], "1")
            self.assertEqual(person[person.index("--epochs") + 1], "35")

    def test_only_person_can_change_amp_recipe_and_other_hyperparameters_remain_locked(self):
        saved = {"amp": True, "seed": 1, "batch_size": 64, "learning_rate": .0125,
                 "r1_fixed_loss_scale": 4096., "git": {"commit": "old"}}
        changed = dict(saved, amp=False, r1_fixed_loss_scale=None, git={"commit": "new"})
        allowed = recovery.permitted_config_delta(saved, changed, "person", person_precision="fp32")
        self.assertIn("amp", allowed)
        for view in ("full", "face"):
            with self.assertRaises(ValueError):
                recovery.permitted_config_delta(saved, changed, view, person_precision="fp32")
        for key, value in (("seed", 2), ("batch_size", 128), ("learning_rate", .001), ("residual_cap", .01)):
            with self.assertRaises(ValueError):
                recovery.permitted_config_delta(saved, dict(changed, **{key: value}), "person", person_precision="fp32")
        with self.assertRaises(ValueError):
            recovery.permitted_config_delta(saved, changed, "person", person_precision="amp")

    def test_fp32_resume_drops_only_person_scaler_and_retains_adam_rng_source(self):
        manifest = {"person_precision": "fp32", "person_loss_scale_after_boundary": None}
        for view in recovery.VIEWS:
            branch = old_contracts.tiny_whole(3, 13)["branches"][view]
            original = branch["training"]
            before = copy.deepcopy(branch)
            model = SimpleNamespace(r1_view=view, r1_resume=original,
                                    r1_fixed_loss_scale=4096., r1_config={"amp": view != "person"})
            seen = []
            def train(model, *args, **kwargs):
                self.assertIsNot(model.r1_resume, original)
                expected = copy.deepcopy(before["training"])
                if view == "person": expected["scaler"] = {}
                self.assertNestedEqual(expected, model.r1_resume)
                self.assertIsNone(model.r1_fixed_loss_scale)
                seen.append(True)
                return "ok"
            with patch.object(recovery, "read", return_value=manifest), \
                 patch.object(recovery.base, "branch_flags", return_value=[]), \
                 patch.object(recovery.runner, "train_task", train), \
                 patch.object(recovery.R1Receiver, "load_own", lambda *a, **kw: None):
                recovery.configure_runtime(Path("test_recovery"))
                self.assertEqual(recovery.runner.train_task(model), "ok")
            self.assertTrue(seen)
            self.assertIs(model.r1_resume, original)
            self.assertEqual(model.r1_fixed_loss_scale, 4096.)
            self.assertNestedEqual(before, branch)

    def test_disabled_scaler_cannot_reenable_from_old_amp_state(self):
        old_state = old_contracts.tiny_whole()["branches"]["person"]["training"]["scaler"]
        before = copy.deepcopy(old_state)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        scaler.load_state_dict(old_state)
        self.assertFalse(scaler.is_enabled())
        self.assertEqual(scaler.state_dict(), {})
        self.assertEqual(scaler.get_scale(), 1.)
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        loss = parameter.square().sum()
        self.assertIs(scaler.scale(loss), loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(int(optimizer.state[parameter]["step"]), 1)
        self.assertFalse(torch.equal(parameter, torch.tensor([2.])))
        self.assertNestedEqual(before, old_state)

    def test_probe_accepts_only_requested_epoch_with_full_updates_and_fp32(self):
        recovery.validate_probe_state(probe_state(), 3, 14, 156)
        for invalid in ("task", "epoch", "updates", "skip", "amp", "scaler", "half", "nonfinite", "adam", "loss"):
            state = probe_state()
            if invalid == "task": state["task"] = 4
            elif invalid == "epoch": state["training"]["next_epoch"] = 15
            elif invalid == "updates": state["training"]["history"][-1]["optimizer_steps"] = 149
            elif invalid == "skip": state["training"]["history"][-1]["skipped_optimizer_steps"] = 7
            elif invalid == "amp": state["config"]["amp"] = True
            elif invalid == "scaler": state["training"]["scaler"] = {"scale": 32.}
            elif invalid == "half": state["model"]["task_selector"] = state["model"]["task_selector"].half()
            elif invalid == "nonfinite": state["model"]["task_selector"][0] = float("nan")
            elif invalid == "adam": state["training"]["optimizer"]["state"][0]["exp_avg"][0] = float("inf")
            else: state["training"]["history"][-1]["current_loss"] = float("nan")
            with self.assertRaises((ValueError, AssertionError, FloatingPointError)):
                recovery.validate_probe_state(state, 3, 14, 156)

    def test_prepare_uses_fp32_scan_and_records_precision_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, root = Path(directory)/"source", Path(directory)/"new"
            boundary = source/"recovery/task3/epoch13.pth"
            boundary.parent.mkdir(parents=True)
            whole = old_contracts.tiny_whole(3, 13)
            torch.save(whole, boundary)
            digest = recovery.sha(boundary)
            diagnostic = Path(directory)/"diagnostic.json"
            row = {"status": "complete", "all_expected_batches_scanned": True,
                   "source_sha256": digest, "view": "person",
                   "mode_nonfinite_batches": {"amp32": 7, "amp4096": 7, "fp32": 0},
                   "source_sha_unchanged": True, "model_unchanged": True, "optimizer_unchanged": True}
            recovery.write(diagnostic, row)
            with patch.object(recovery, "BOUNDARY_SHA", digest), \
                 patch.object(recovery, "prefix_paths", return_value=[boundary]), \
                 patch.object(recovery.runner, "git_metadata", return_value={"head": "test"}):
                manifest = recovery.prepare(source, root, diagnostic, 32., person_precision="fp32")
            self.assertEqual(manifest["person_precision"], "fp32")
            self.assertIsNone(manifest["person_loss_scale_after_boundary"])
            self.assertEqual(recovery.sha(boundary), digest)
            self.assertNestedEqual(torch.load(root/"recovery/task3/epoch13.pth", map_location="cpu"), whole)
            # A successful AMP scan cannot substitute for a failed FP32 scan.
            row["mode_nonfinite_batches"] = {"amp32": 0, "amp4096": 0, "fp32": 1}
            recovery.write(diagnostic, row)
            bad_root = Path(directory)/"bad"
            with patch.object(recovery, "BOUNDARY_SHA", digest), self.assertRaises(ValueError):
                recovery.prepare(source, bad_root, diagnostic, 32., person_precision="fp32")
            self.assertFalse(bad_root.exists())

    def test_nonfinite_gradient_never_reaches_adam_or_changes_parameters(self):
        for bad in (float("nan"), float("inf")):
            parameter = torch.nn.Parameter(torch.tensor([2.]))
            optimizer = torch.optim.Adam([parameter], lr=.01)
            parameter.square().sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.tensor([bad])
            before_param = parameter.detach().clone()
            before_state = copy.deepcopy(optimizer.state_dict())
            calls = []
            def observed_step(opt):
                calls.append(True)
                return torch.optim.Adam.step(opt)
            with self.assertRaises(FloatingPointError):
                recovery.guarded_adam_step(observed_step, optimizer)
            self.assertEqual(calls, [])
            self.assertTrue(torch.equal(parameter, before_param))
            self.assertNestedEqual(before_state, optimizer.state_dict())

    def test_finite_gradient_updates_adam_once_and_rejects_nonfinite_post_state(self):
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        parameter.square().sum().backward()
        recovery.guarded_adam_step(torch.optim.Adam.step, optimizer)
        self.assertEqual(int(optimizer.state[parameter]["step"]), 1)
        self.assertTrue(torch.isfinite(parameter).all())
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        def corrupted_adam(opt):
            torch.optim.Adam.step(opt)
            opt.state[parameter]["exp_avg"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            recovery.guarded_adam_step(corrupted_adam, optimizer)

    def test_bad_loss_stops_before_backward_and_guard_restores_runtime(self):
        original_step = torch.optim.Adam.step
        calls = []
        def backward(*args, **kwargs):
            calls.append(True)
        def train(model):
            return recovery.runner.backward_routed_training_losses(
                model_loss=torch.tensor(float("nan")), adapter_loss=torch.tensor(.2))
        with patch.object(recovery.runner, "backward_routed_training_losses", backward):
            with self.assertRaises(FloatingPointError):
                recovery.guarded_fp32_train(train, object())
            self.assertIs(recovery.runner.backward_routed_training_losses, backward)
        self.assertEqual(calls, [])
        self.assertIs(torch.optim.Adam.step, original_step)

    def test_probe_stops_at_epoch14_and_never_reports_failed_updates_as_success(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                branch = recovery.branch_paths(root)["person"]
                previous_path = branch/"epoch_recovery/task3/epoch13.pth"
                target = branch/"epoch_recovery/task3/epoch14.pth"
                previous_path.parent.mkdir(parents=True)
                previous = old_contracts.tiny_whole(3, 13)["branches"]["person"]
                previous["training"]["optimizer"]["state"][0]["step"] = torch.tensor(2028.)
                torch.save(previous, previous_path)
                manifest = {"person_precision": "fp32", "smoke": False,
                            "task": 3, "completed_epoch": 13}
                events = []
                def worker(output, view, smoke, condition):
                    self.assertEqual((output, view, smoke, condition), (root, "person", False, "full_extend"))
                    saved = probe_state()
                    saved["training"]["optimizer"]["state"][0]["step"] = torch.tensor(2184.)
                    if failed:
                        saved["training"]["history"][-1]["skipped_optimizer_steps"] = 7
                    torch.save(saved, target)
                    events.append("saved_epoch14")
                    recovery.base.wait_for(root/"barriers/task3/epoch14_accepted.json", root)
                    events.append("unexpected_epoch15")
                original_wait = recovery.base.wait_for
                with patch.object(recovery, "verify_prefix", return_value=manifest), \
                     patch.object(recovery, "configure_runtime", return_value=None), \
                     patch.object(recovery.base, "worker", worker):
                    if failed:
                        with self.assertRaises(ValueError):
                            recovery.probe_person(root)
                    else:
                        recovery.probe_person(root)
                self.assertEqual(events, ["saved_epoch14"])
                self.assertIs(recovery.base.wait_for, original_wait)
                result_path = root/"person_epoch_probe.json"
                self.assertEqual(result_path.exists(), not failed)
                if not failed:
                    result = recovery.read(result_path)
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["effective_updates"], 156)
                    self.assertEqual(result["skipped_updates"], 0)
                    self.assertEqual(result["epoch"], 14)


if __name__ == "__main__":
    unittest.main()
