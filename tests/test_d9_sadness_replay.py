import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from scripts.emotic.d9_sadness_replay import verify_metrics, replay
from scripts.emotic.fit_d9_task10_probe import scores


class SadnessReplayTests(unittest.TestCase):
    def test_comparison_rejects_an_unrelated_training_protocol(self):
        with TemporaryDirectory() as tmp:
            tcp = Path(tmp)
            (tcp/'probe').mkdir()
            (tcp/'probe'/'tcp_validation_lock.json').write_text(
                json.dumps({'protocol': {'source_mode': 'sample_full'}}))
            with self.assertRaisesRegex(ValueError, 'data protocol'):
                replay(tcp, tcp, 6, 'val', source_mode='sample_holdout')

    def test_per_class_replay_checks_both_availability_groups(self):
        z = torch.tensor([[0., 2.], [1., 0.], [2., 3.], [0., -1.]])
        y = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        face = np.array([True, False, True, False])
        masks = {'all': np.ones(4, dtype=bool), 'reliable': face, 'unreliable': ~face}
        row = {'metrics': {name: scores(z[mask], y[mask]) for name, mask in masks.items()}}
        verify_metrics(z, y, face, row)
        row['metrics']['reliable']['per_class_ap'][1] -= 1.
        with self.assertRaises(AssertionError):
            verify_metrics(z, y, face, row)
