import unittest

import numpy as np

from scripts.emotic.run_r1_frozen9_learned_fusion import (
    FIXED, Rows, blend, fit_task, stage_logits,
)


def example_rows():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    score = 4 * (2 * targets - 1)
    # Face is best on reliable people. Person is best on the others.
    reliable = np.array([True, True, False, False])
    logits = np.stack((-score, np.where(reliable[:, None], -score, score), score), axis=1)
    return Rows(np.array(["train:a", "train:b", "train:c", "train:d"]), logits, targets, reliable)


class Frozen9LearnedFusionTests(unittest.TestCase):
    def test_separate_fits_recover_each_condition_and_keep_ablation_population_exact(self):
        rows = example_rows()
        weights, report = fit_task(rows)
        np.testing.assert_allclose(weights["independent"], [[0, 1, 0], [0, 0, 1]], atol=1e-7)
        fixed = blend(rows.logits, rows.reliable, FIXED)
        rel_only = blend(rows.logits, rows.reliable, weights["reliable_only"])
        unrel_only = blend(rows.logits, rows.reliable, weights["unreliable_only"])
        np.testing.assert_array_equal(fixed[~rows.reliable], rel_only[~rows.reliable])
        np.testing.assert_array_equal(fixed[rows.reliable], unrel_only[rows.reliable])
        assert report["reliable"]["train_bce"]["independent"] < report["reliable"]["train_bce"]["fixed"]
        assert report["unreliable"]["train_bce"]["independent"] < report["unreliable"]["train_bce"]["fixed"]


    def test_empty_condition_uses_original_weights(self):
        rows = example_rows()
        rows.reliable[:] = False
        weights, report = fit_task(rows)
        assert report["reliable"]["fallback_empty_group"]
        for matrix in weights.values():
            np.testing.assert_array_equal(matrix[1], FIXED[1])


    def test_stage_uses_old_fusion_for_old_class_columns(self):
        rows = example_rows()
        early = Rows(rows.ids, np.tile(rows.logits[:, :, :1], (1, 1, 5)),
                     np.tile(rows.targets[:, :1], (1, 5)), rows.reliable)
        later = Rows(rows.ids, np.tile(rows.logits[:, :, :1], (1, 1, 8)),
                     np.tile(rows.targets[:, :1], (1, 8)), rows.reliable)
        past, _ = fit_task(rows)
        bank = {0: past}
        original = stage_logits(early, bank, 0, "independent")
        bank[1] = {"independent": FIXED.copy()}
        result = stage_logits(later, bank, 1, "independent")
        np.testing.assert_array_equal(result[:, :5], original)
        np.testing.assert_array_equal(result[:, 5:], blend(later.logits[:, :, 5:], rows.reliable, FIXED))
