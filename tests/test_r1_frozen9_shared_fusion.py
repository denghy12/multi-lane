import unittest
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.runner import task_indices
from scripts.emotic.run_r1_frozen9_fusion_ap import candidate_map
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, Rows, blend, stage_logits
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import image_fold, image_key
from scripts.emotic.run_r1_frozen9_shared_fusion import (
    EQUAL, GRID, ChronologicalFit, fit_smooth_ap, fused_scores, grid_utility, rank_support, weight_matrices,
)


class SharedFusionTests(unittest.TestCase):
    def test_two_parameters_represent_equal_and_fixed_weights(self):
        np.testing.assert_array_equal(weight_matrices((.8, .2)), FIXED)
        equal = weight_matrices(EQUAL)
        np.testing.assert_array_equal(equal[0], [.5, .5, 0])
        np.testing.assert_array_equal(equal[1], np.full(3, 1 / 3, dtype=np.float32))
        matrices = weight_matrices(GRID)
        np.testing.assert_allclose(matrices.sum(axis=-1), 1, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(matrices[:, 0, 2], 0)

    def test_grid_utility_matches_reference_scoring(self):
        rng = np.random.default_rng(31)
        logits = rng.normal(size=(11, 3, 2)).astype(np.float32)
        target = (rng.random((11, 2)) < .4).astype(np.float32)
        reliable = np.arange(11) % 2 == 0
        utility = grid_utility(logits, target, reliable)
        candidates = (0, 77, 1320, len(GRID) - 1)
        expected = [candidate_map(blend(logits, reliable, weight_matrices(GRID[index]))[None], target)[0]
                    for index in candidates]
        np.testing.assert_array_equal(utility[list(candidates)], expected)

    def test_smooth_ap_updates_and_face_mask_is_exact(self):
        labels = torch.tensor([[1.], [0.], [1.], [0.]])
        logits = torch.tensor([[[-2.], [-1.], [3.]], [[2.], [1.], [-3.]],
                               [[-2.], [2.], [-10.]], [[2.], [-2.], [10.]]])
        reliable = torch.tensor([True, True, False, False])
        changed = logits.clone()
        changed[~reliable, 2] += 100
        initial = torch.tensor(EQUAL, dtype=torch.float32).logit()
        torch.testing.assert_close(fused_scores(initial.sigmoid(), changed, reliable),
                                   fused_scores(initial.sigmoid(), logits, reliable), atol=0, rtol=0)
        task = (logits, reliable, rank_support(labels))
        once, loss = fit_smooth_ap([task], initial, 10)
        twice, _ = fit_smooth_ap([task, task], initial, 10)
        self.assertGreater(loss[0]["gradient_norm"], 0)
        self.assertLess(loss[-1]["objective"], loss[0]["objective"])
        torch.testing.assert_close(once, twice)

    def test_fitter_uses_only_each_folds_available_prefix_and_freezes_old_blocks(self):
        ids = np.array([f"val:img{i}#person=0" for i in range(40)])
        folds = np.array([image_fold(image_key(person)) for person in ids])
        reliable = np.arange(len(ids)) % 2 == 0
        fit = ChronologicalFit("cpu", steps=2)
        bank = {}
        previous = None
        for task in (0, 1):
            columns = list(task_indices(task))
            n = columns[-1] + 1
            logits = np.arange(len(ids) * 3 * n, dtype=np.float32).reshape(len(ids), 3, n) / 50
            targets = np.tile((np.arange(len(ids)) % 3 == 0)[:, None], (1, n)).astype(np.float32)
            rows = Rows(ids, logits, targets, reliable)
            for fold in (0, 1):
                mask = folds != fold
                if fold == 0 and task == 0:
                    with self.assertRaisesRegex(AssertionError, "order"):
                        fit(rows, mask, list(task_indices(1)))
                with patch("scripts.emotic.run_r1_frozen9_shared_fusion.fit_controls",
                           return_value=({"fixed": FIXED, "bounded": FIXED}, {})):
                    weights, support = fit(rows, mask, columns)
                self.assertEqual(support["shared"]["prefix_tasks"], list(range(task + 1)))
                torch.testing.assert_close(fit.histories[fold].cohorts[-1][0],
                                           torch.from_numpy(logits[mask][:, :, columns]))
                if fold == 0:
                    bank[task] = weights
            if task == 0:
                previous = bank[0]["prefix_smooth_ap"].copy()
                np.testing.assert_array_equal(bank[0]["frozen_smooth_ap"], previous)
            else:
                np.testing.assert_array_equal(bank[1]["frozen_smooth_ap"], previous)
                np.testing.assert_array_equal(bank[0]["prefix_smooth_ap"], previous)
                expected = blend(logits[:, :, list(task_indices(0))], reliable, previous)
                np.testing.assert_array_equal(stage_logits(rows, bank, 1, "prefix_smooth_ap")[:, :len(task_indices(0))],
                                              expected)
