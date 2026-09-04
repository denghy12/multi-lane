import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import test_track_a_validation_ensemble_control as fixtures
from multi_lane.track_a.compare_fixed_test_ensembles import (
    MAPPING, RULE, audit_test_run, compare_fixed_test_ensembles,
)
from multi_lane.track_a.compare_validation_ensembles import compare_ensembles
from multi_lane.track_a.runner import compute_metrics


class FixedTestEnsembleTest(unittest.TestCase):
    @staticmethod
    def make_runs(root):
        full, person = fixtures.ValidationEnsembleTest.make_runs(root)
        reference = root / 'validation_reference.json'
        reference.write_text(json.dumps({'evaluation_split': 'val', 'search_performed': False,
                                         'rule': RULE, 'auxiliary_seed_mapping': MAPPING}))
        for run in full + person:
            config = json.loads((run / 'config.json').read_text())
            config.update(reporting_split='test', evaluation_score_purpose='fixed_test_fusion')
            summary = json.loads((run / 'seed_summary.json').read_text())
            summary['config'] = config
            (run / 'config.json').write_text(json.dumps(config))
            (run / 'seed_summary.json').write_text(json.dumps(summary))
            (run / 'validation_scores').rename(run / 'test_scores')
            for path in (run / 'test_scores').glob('*.npz'):
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays['sample_ids'] = np.char.replace(arrays['sample_ids'].astype(str), 'val:', 'test:')
                np.savez_compressed(path, **arrays)
        return full, person, reference

    def test_fixed_mapping_probabilities_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person, reference = self.make_runs(Path(directory))
            result = compare_fixed_test_ensembles(full[::-1], person[::-1], reference)
            self.assertEqual(result['evaluation_split'], 'test')
            self.assertFalse(result['search_performed_on_test'])
            self.assertFalse(result['training_performed'])
            self.assertNotIn('winner', result)
            self.assertEqual(result['auxiliary_seed_mapping'], MAPPING)
            self.assertEqual(len(result['sources'][0]['artifact_sha256']), 11)
            for row in result['groups']['cyclic_full_full']:
                seed, auxiliary = row['seed'], row['auxiliary_seed']
                first = audit_test_run(full[seed], 'full')[1][-1]
                second = audit_test_run(full[auxiliary], 'full')[1][-1]
                expected = compute_metrics(7, torch.from_numpy(0.8 * first.probabilities + 0.2 * second.probabilities),
                                           torch.from_numpy(first.targets), 0.5)
                self.assertEqual(row['metrics']['final_mAP'], expected.mAP)

    def test_validation_entry_still_rejects_test(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person, _ = self.make_runs(Path(directory))
            with self.assertRaisesRegex(ValueError, 'test is forbidden'):
                compare_ensembles(full, person)

    def test_test_entry_rejects_validation_and_changed_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person, reference = self.make_runs(Path(directory))
            changed = json.loads(reference.read_text())
            changed['rule']['auxiliary_weight'] = 0.25
            reference.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'predeclared'):
                compare_fixed_test_ensembles(full, person, reference)
            config = json.loads((full[0] / 'config.json').read_text())
            config['reporting_split'] = 'val'
            (full[0] / 'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'held-out test'):
                audit_test_run(full[0], 'full')

    def test_rejects_duplicate_seed_and_corrupted_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person, reference = self.make_runs(Path(directory))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                compare_fixed_test_ensembles([full[0], full[0], full[2]], person, reference)
            summary = json.loads((full[0] / 'seed_summary.json').read_text())
            summary['task_metrics'][0]['per_class_ap'][0] -= 1
            (full[0] / 'seed_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'per_class_ap'):
                audit_test_run(full[0], 'full')

    def test_rejects_wrong_split_ids_and_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            full, _, _ = self.make_runs(Path(directory))
            path = full[0] / 'test_scores' / 'task0.npz'
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays['sample_ids'] = np.char.replace(arrays['sample_ids'].astype(str), 'test:', 'val:')
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'Non-test sample'):
                audit_test_run(full[0], 'full')
            summary = json.loads((full[1] / 'seed_summary.json').read_text())
            summary['completed_epochs'] = 239
            (full[1] / 'seed_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'training budget'):
                audit_test_run(full[1], 'full')


if __name__ == '__main__':
    unittest.main()
