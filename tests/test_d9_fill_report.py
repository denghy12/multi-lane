"""Report incomplete experiments as incomplete and reject broken evaluation seals."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.emotic.d9_fill_report import collect, digest, markdown, read_eight_pair


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class TestFillReport(unittest.TestCase):
    def test_empty_outputs_preserve_historical_and_mark_all_missing(self):
        with tempfile.TemporaryDirectory() as d:
            report = collect(Path(d))
        self.assertFalse(report['complete'])
        self.assertEqual(report['expected_missing_cells'], 21)
        self.assertEqual(report['newly_completed_cells'], 0)
        self.assertEqual(report['rows']['late_bce']['results']['equal90']['test'], [31.5094, 38.7883])
        self.assertNotIn('equal90', report['rows']['global_relation']['results'])
        self.assertIn('待完成', markdown(report))
        self.assertEqual(report['rows']['tcp_fixed']['scope'], 'task6_current')

    def test_replay_requires_both_stage_count_and_exact_audit(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            val = folder / 'evaluation/val/result.json'
            test = folder / 'evaluation/test/result.json'
            write(val, dict(split='val', smoke=False, stages=[{}] * 8, summary={}))
            write(test, dict(split='test', smoke=False, summary={}, stages=[
                {'protocol': {'validation_result_sha256': digest(val)}}] * 8))
            self.assertIsNone(read_eight_pair(folder, 'replay'))
            for split in ('val', 'test'):
                write(folder / 'branch_replay' / f'{split}_predictions.json',
                      dict(status='exact', stages=[{'status': 'exact'}] * 8))
            self.assertIsNotNone(read_eight_pair(folder, 'replay'))
            write(test, dict(split='test', smoke=False, stages=[{}] * 3))
            with self.assertRaisesRegex(ValueError, 'eight-stage'):
                read_eight_pair(folder, 'replay')

    def test_changed_validation_cannot_publish_test(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            val = folder / 'evaluation/val/result.json'
            test = folder / 'evaluation/test/result.json'
            write(val, dict(split='val', smoke=False, stages=[{}] * 8, summary={}))
            write(test, dict(split='test', smoke=False, summary={}, stages=[
                {'protocol': {'validation_result_sha256': 'different'}}] * 8))
            with self.assertRaisesRegex(ValueError, 'different Validation'):
                read_eight_pair(folder, 'replay')


if __name__ == '__main__':
    unittest.main()
