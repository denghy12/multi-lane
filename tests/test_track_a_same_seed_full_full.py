import json
import shutil
import tempfile
import unittest
from pathlib import Path

import test_track_a_fixed_test_ensemble_control as fixtures
from multi_lane.track_a.compare_same_seed_full_full import compare_same_seed


class SameSeedFullTest(unittest.TestCase):
    @staticmethod
    def make_runs(root):
        full, person, _ = fixtures.FixedTestEnsembleTest.make_runs(root)
        repeat = []
        for seed, run in enumerate(full):
            path = root / f'repeat{seed}'
            shutil.copytree(run, path)
            repeat.append(path)
        return full, repeat, person

    def test_identical_repeat_diagnostics_and_same_seed_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            result = compare_same_seed(*self.make_runs(Path(directory)))
            self.assertEqual(result['auxiliary_seed_mapping'], [[0,0],[1,1],[2,2]])
            self.assertFalse(result['search_performed_on_test'])
            self.assertEqual(result['rule']['auxiliary_weight'], 0.2)
            for record in result['repeat_diagnostics']:
                self.assertTrue(record['training_history_exact_except_elapsed'])
                self.assertTrue(all(row['differing_probability_count'] == 0 for row in record['tasks']))

    def test_rejects_same_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            full, _, person = self.make_runs(Path(directory))
            with self.assertRaisesRegex(ValueError, 'distinct training run'):
                compare_same_seed(full, full, person)

    def test_rejects_repeat_config_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            full, repeat, person = self.make_runs(Path(directory))
            config = json.loads((repeat[0]/'config.json').read_text())
            config['learning_rate'] = 0.99
            summary = json.loads((repeat[0]/'seed_summary.json').read_text())
            summary['config'] = config
            (repeat[0]/'config.json').write_text(json.dumps(config))
            (repeat[0]/'seed_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'identical config'):
                compare_same_seed(full, repeat, person)


if __name__ == '__main__':
    unittest.main()
