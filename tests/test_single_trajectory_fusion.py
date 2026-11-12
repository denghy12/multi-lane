import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.single_trajectory_fusion import (
    OnlineFusionSchedules, isolated_observation, update_budget,
)
from multi_lane.track_a.r1_crossvit import atomic_save
from scripts.emotic.run_r1_single_trajectory_fusion import complete_checkpoint, branch_paths


class SingleTrajectoryTests(unittest.TestCase):
    def test_complete_epochs_keep_immutable_recovery_and_latest_pointer(self):
        a = OnlineFusionSchedules((2,), smoke=True)
        a.begin_task(0)
        with tempfile.TemporaryDirectory() as folder, patch(
                "scripts.emotic.run_r1_single_trajectory_fusion.runner.git_metadata", return_value={}):
            root = Path(folder)
            for epoch in (1, 2):
                for path in branch_paths(root).values():
                    atomic_save({"task": 0, "model": {"p": torch.tensor(epoch)},
                        "training": {"next_epoch": epoch, "completed_task_updates": epoch*2}},
                        path/"epoch_recovery/task0"/f"epoch{epoch:02d}.pth")
                complete_checkpoint(root, 0, epoch, a, "same_run")
            self.assertTrue((root/"recovery/task0/epoch01.pth").exists())
            self.assertTrue((root/"recovery/task0/epoch02.pth").exists())
            self.assertIn('"epoch": 2', (root/"checkpoints/latest.json").read_text())

    def test_budgets_match_and_mid_freeze_is_exact(self):
        for smoke, epochs, expected in ((False, 30, 100), (True, 3, 2)):
            for method in ("continuous", "mid_freeze"):
                self.assertEqual(sum(update_budget(method, e, smoke) for e in range(1, epochs+1)), expected)
        self.assertEqual(update_budget("mid_freeze", 16), 0)
        self.assertEqual(update_budget("continuous", 30), 4)

    def rows(self, classes):
        generator = torch.Generator().manual_seed(42)
        return ([f"train:image{i}#person=0" for i in range(8)],
            torch.randn(8, 3, classes, generator=generator, requires_grad=True),
            torch.randint(0, 2, (8, classes), generator=generator).float(),
            torch.tensor([True, False]*4))

    def test_detached_gradient_mask_and_freezing_old_tasks(self):
        schedules = OnlineFusionSchedules((2, 1), smoke=True)
        schedules.begin_task(0)
        rows = self.rows(2)
        schedules.warmup(1)
        report = schedules.observe(2, *rows)
        self.assertTrue(any(g > 0 for g in report["continuous"]["gradient_norms"]))
        self.assertIsNone(rows[1].grad)
        mid = schedules.banks["mid_freeze"].tasks[0].raw_shares.clone()
        schedules.observe(3, *rows)
        self.assertTrue(torch.equal(mid, schedules.banks["mid_freeze"].tasks[0].raw_shares))
        bank = schedules.banks["continuous"]
        with torch.no_grad():
            z, good = rows[1], rows[3]
            expected = bank(z, good, 0)
            changed = z.clone()
            changed[~good, 2] += 10000
            self.assertTrue(torch.equal(expected[~good], bank(changed, good, 0)[~good]))
        old = copy.deepcopy(bank.state_dict())
        schedules.begin_task(1)
        schedules.warmup(1)
        schedules.observe(2, *self.rows(1))
        self.assertTrue(all(torch.equal(v, bank.state_dict()[k]) for k, v in old.items()))
        self.assertFalse(bank.tasks[0].raw_shares.requires_grad)

    def test_roundtrip_optimizer_and_schedule(self):
        a = OnlineFusionSchedules((2,), smoke=True)
        a.begin_task(0)
        a.warmup(1)
        rows = self.rows(2)
        a.observe(2, *rows)
        b = OnlineFusionSchedules((2,), smoke=True)
        b.load_state_dict(a.state_dict())
        a.observe(3, *rows)
        b.observe(3, *rows)
        for name in a.banks:
            self.assertTrue(all(torch.equal(v, b.banks[name].state_dict()[k])
                for k, v in a.banks[name].state_dict().items()))
        with self.assertRaises(ValueError):
            b.observe(3, *rows)

    def test_validation_cannot_update_fusion(self):
        a = OnlineFusionSchedules((2,), smoke=True)
        a.begin_task(0)
        a.warmup(1)
        ids, *rest = self.rows(2)
        with self.assertRaises(ValueError):
            a.observe(2, [s.replace("train:", "val:") for s in ids], *rest)

    def test_observation_restores_rng_modes_and_flags_even_on_error(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.Dropout())
        model[0].eval()
        model[0].weight.requires_grad_(False)
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        modes = [m.training for m in model.modules()]
        flags = [p.requires_grad for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with isolated_observation(model):
                random.random(); np.random.rand(); torch.rand(7)
                model.requires_grad_(True)
                raise RuntimeError("probe")
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])


if __name__ == "__main__":
    unittest.main()
