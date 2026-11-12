import unittest
import numpy as np
from scripts.emotic.r1_full_train_fusion_weights import locked_weights, task_blend, align_exact


def report():
    return {str(t): {'heldout_bce': {'trace': [{'step': 99, 'shares': [.8, .2]}]}}
            for t in range(8)}


class RealReplayTests(unittest.TestCase):
    def test_locked_state_rejects_missing_ambiguous_or_invalid_shares(self):
        for change in ('missing', 'ambiguous', 'invalid'):
            rows = report()
            trace = rows['7']['heldout_bce']['trace']
            if change == 'missing':
                trace.clear()
            elif change == 'ambiguous':
                trace.append(trace[0])
            else:
                trace[0]['shares'] = [float('nan'), .2]
            with self.assertRaises(ValueError):
                locked_weights(rows)

    def test_alignment_reorders_and_checks_labels_and_duplicate_people(self):
        ids = np.array(['a', 'b'])
        targets = np.array([[1], [0]])
        z = np.array([[7], [3]])
        np.testing.assert_array_equal(align_exact(ids, targets, ids[::-1], targets[::-1], z), z[::-1])
        with self.assertRaises(ValueError):
            align_exact(ids, targets, ids, targets[::-1], z)
        with self.assertRaises(ValueError):
            align_exact(ids, targets, np.array(['a', 'a']), targets, z)
        with self.assertRaises(ValueError):
            align_exact(ids, targets, np.array(['a', 'c']), targets, z)

    def test_task_blocks_and_unreliable_face_mask(self):
        weights = locked_weights(report())
        weights[1] = np.array([[.7, .3, 0], [.35, .15, .5]], dtype=np.float32)
        z = np.random.RandomState(3).randn(4, 3, 8).astype(np.float32)
        mask = np.array([True, False, True, False])
        actual = task_blend(z, mask, weights, 1)
        expected = np.concatenate([
            sum(weights[t][mask.astype(int), v, None]*z[:, v, lo:hi] for v in range(3))
            for t, lo, hi in [(0, 0, 5), (1, 5, 8)]
        ], axis=1)
        np.testing.assert_array_equal(actual, expected)
        z[~mask, 2] += 10000
        np.testing.assert_array_equal(task_blend(z, mask, weights, 1)[~mask], actual[~mask])
        with self.assertRaises(ValueError):
            task_blend(z[:, :, :7], mask, weights, 1)
