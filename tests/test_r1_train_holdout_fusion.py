import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from multi_lane.track_a.runner import fit_calibration_indices
from scripts.emotic.run_r1_train_holdout_fusion import (
    fit_masks, heldout_id, image_key, training_command,
)


class TrainHoldoutFusionTests(unittest.TestCase):
    def test_image_partition_matches_actual_branch_exclusion(self):
        ids = np.array([f"train:image{i}.jpg#person={p}" for i in range(120) for p in range(2)])
        source = SimpleNamespace(sample_ids=ids.tolist(), targets=[(0,)]*len(ids))
        fit, heldout = fit_calibration_indices(source, (0,), .2)
        actual = np.array([heldout_id(s) for s in ids])
        self.assertEqual(set(np.flatnonzero(actual)), set(heldout))
        self.assertEqual(set(np.flatnonzero(~actual)), set(fit))
        seen_control, fusion = fit_masks(ids)
        np.testing.assert_array_equal(fusion, actual)
        self.assertFalse(bool((seen_control & fusion).any()))
        self.assertLessEqual(abs(int(seen_control.sum())-int(fusion.sum())), 2)
        self.assertFalse(set(map(image_key, ids[seen_control])).intersection(map(image_key, ids[fusion])))

    def test_partition_is_identical_across_views_and_task_subsets(self):
        ids = [f"train:image{i}#person=0" for i in range(60)]
        expected = {s: heldout_id(s) for s in ids}
        for indices in (range(10, 50), range(20, 60), range(0, 40)):
            self.assertEqual([heldout_id(ids[i]) for i in indices], [expected[ids[i]] for i in indices])
        with self.assertRaises(ValueError):
            heldout_id("val:image0#person=0")

    def test_commands_declare_exclusion_and_keep_cross_view_direction(self):
        def fake_flags(*args, **kwargs):
            return ["--reporting-split", "test", "--evaluation-score-purpose", "fixed_test_fusion"]
        with patch("scripts.emotic.run_r1_train_holdout_fusion.flags", side_effect=fake_flags):
            full = training_command(Path("output/study"), "full", False)
            person = training_command(Path("output/study"), "person", False)
            face = training_command(Path("output/study"), "face", False)
        for command in (full, person, face):
            self.assertEqual(command[command.index("--calibration-fraction")+1], ".2")
            self.assertIn("--train-only-fusion-holdout", command)
            self.assertEqual(command[command.index("--reporting-split")+1], "val")
        self.assertNotIn("--r1-frozen-mode", full)
        for command in (person, face):
            self.assertEqual(command[command.index("--r1-exchange-layer")+1], "8")
            self.assertEqual(command[command.index("--r1-frozen-evidence")+1], "all")
