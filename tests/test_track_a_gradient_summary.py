import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_gradient_attribution import summarize


class GradientSummaryTest(unittest.TestCase):
    def test_missing_gradients_do_not_count_as_valid_or_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'config.json').write_text(json.dumps({'diagnostic_git': {'commit': 'fixture'}}))
            (root/'result.json').write_text(json.dumps({'status': 'complete', 'training_updates': 0, 'snapshots': 4}))
            for batch, cosine in enumerate([-.2, -.3, -.4, None]):
                record = dict(method='baseline', task=0, precision='deterministic_fp32', batch=batch,
                              objective_names=['view_full', 'view_person'], geometry={
                                  'selector': dict(norms=[1, 1 if cosine is not None else None],
                                                   cosines=[[1, cosine], [cosine, 1]])})
                (root/f'baseline_task0_deterministic_fp32_batch{batch}.json').write_text(json.dumps(record))
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root)
            summary = json.loads((root/'summary.json').read_text())
            pair = next(x for x in summary['aggregates'] if x['metric']=='cosine')
            self.assertEqual((pair['n'], pair['missing'], pair['below_minus_point_one']), (3, 1, 3))
            self.assertEqual(summary['descriptive_stability'][0]['stable_tasks'], [])
            (root/'result.json').write_text(json.dumps({'status': 'complete', 'training_updates': 0, 'snapshots': 5}))
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
