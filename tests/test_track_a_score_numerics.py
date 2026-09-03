import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.evaluation_scores import (
    align_evaluation_scores, batched_sigmoid, load_evaluation_scores,
)
from multi_lane.track_a.runner import evaluate, write_evaluation_scores


class ScoreNumericsTest(unittest.TestCase):
    def test_legacy_reconstruction_requires_batch_size(self):
        with self.assertRaisesRegex(ValueError, 'eval_batch_size'):
            batched_sigmoid(np.zeros((65, 2), dtype=np.float32), None)

    def test_legacy_uses_original_batches_not_whole_array_kernel(self):
        # Model a shape-dependent one-ULP difference observed on seed1/task5.
        # The mock keeps this regression portable across CPU vector kernels.
        logits = np.zeros((66, 2), dtype=np.float32)
        original = torch.sigmoid

        def shape_sensitive_sigmoid(value):
            probabilities = original(value)
            if value.shape[0] > 64:
                probabilities[-1, -1] = torch.nextafter(probabilities[-1, -1], torch.tensor(1.0))
            return probabilities

        with patch('torch.sigmoid', side_effect=shape_sensitive_sigmoid) as operation:
            reconstructed = batched_sigmoid(logits, 64)
            self.assertEqual([call.args[0].shape[0] for call in operation.call_args_list], [64, 2])
            whole = torch.sigmoid(torch.from_numpy(logits)).numpy()
        self.assertNotEqual(whole[-1, -1], reconstructed[-1, -1])
        np.testing.assert_array_equal(reconstructed, np.full((66, 2), 0.5, dtype=np.float32))

    def test_v2_round_trip_uses_saved_probabilities_without_sigmoid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.npz'
            logits = torch.zeros(3, 2)
            probabilities = torch.full((3, 2), 0.5)
            probabilities[-1, -1] = torch.nextafter(probabilities[-1, -1], torch.tensor(1.0))
            write_evaluation_scores(path, 0, ['a', 'b', 'c'], logits, torch.zeros_like(logits), probabilities, [2, 1])
            with patch('torch.sigmoid', side_effect=AssertionError('Stored probabilities must not be recalculated')):
                dump = load_evaluation_scores(path)
            np.testing.assert_array_equal(dump.probabilities, probabilities.numpy())
            self.assertEqual(dump.probability_source, 'stored_evaluation_probabilities_v2')
            with np.load(path) as data:
                self.assertEqual(int(data['schema_version']), 2)
                self.assertEqual(data['batch_lengths'].tolist(), [2, 1])

    def test_alignment_moves_probabilities_with_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a.npz', Path(directory)/'b.npz'
            logits = torch.tensor([[1.0], [-1.0]])
            targets = torch.tensor([[1.0], [0.0]])
            write_evaluation_scores(a, 0, ['a','b'], logits, targets, torch.sigmoid(logits), [2])
            write_evaluation_scores(b, 0, ['b','a'], logits.flip(0), targets.flip(0), torch.sigmoid(logits).flip(0), [2])
            arrays = align_evaluation_scores(load_evaluation_scores(a), load_evaluation_scores(b))
            np.testing.assert_array_equal(arrays[4], arrays[5])

    def test_writer_rejects_bad_batch_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'batch lengths'):
                write_evaluation_scores(Path(directory)/'bad.npz', 0, ['a','b'], torch.zeros(2,1),
                                        torch.zeros(2,1), torch.full((2,1),0.5), [1])

    def test_evaluate_exports_exact_probabilities_and_partial_final_batch(self):
        class Model(torch.nn.Module):
            def seen_logits(self, images):
                return images
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.npz'
            loader = [(torch.zeros(2, 5), torch.zeros(2, 5), ['a','b']),
                      (torch.ones(1, 5), torch.ones(1, 5), ['c'])]
            metrics = evaluate(Model(), loader, torch.device('cpu'), 0, 0.5, False, path)
            dump = load_evaluation_scores(path)
            expected = torch.cat([torch.sigmoid(batch[0]) for batch in loader]).numpy()
            np.testing.assert_array_equal(dump.probabilities, expected)
            self.assertEqual(metrics.samples, 3)
            with np.load(path) as data:
                self.assertEqual(data['batch_lengths'].tolist(), [2,1])


if __name__ == '__main__':
    unittest.main()
