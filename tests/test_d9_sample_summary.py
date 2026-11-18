import json
import tempfile
import unittest
from pathlib import Path

from scripts.emotic.summarize_d9_sample_fusion import summarize


class SampleSummaryTests(unittest.TestCase):
    def test_full_asl_paired_only_on_identical_replayed_subjects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('sample_full', 'sample_holdout', 'sample_full_asl'):
                for split in ('val', 'test'):
                    path = root / mode / 'smoke_evaluation' / split
                    path.mkdir(parents=True)
                    names = ['fixed', 'equal']
                    if mode != 'sample_full_asl':
                        names.extend(['mid_bce', 'late_bce'])
                    if mode != 'sample_full':
                        names.append('mid_asl')
                    data = {m: {'all': dict(final_mAP=float(m == 'mid_asl'), average_mAP=2.)} for m in names}
                    stages = [{'exact_replay': mode != 'sample_holdout',
                               'protocol': {'reference_scores_sha256': f'{split}-{i}', 'task': i}} for i in range(2)]
                    (path / 'result.json').write_text(json.dumps(dict(
                        mode=mode, split=split, smoke=True, stages=stages, summary=data)))
            extra = root / 'sample_full_asl'
            summarize(root, smoke=True)
            previous = (root / 'comparison.json').read_bytes()
            summarize(root, smoke=True, full_asl_root=extra)
            self.assertEqual((root / 'comparison.json').read_bytes(), previous)
            result = json.loads((extra / 'comparison_with_full_asl.json').read_text())
            delta = result['paired_asl_minus_bce_by_cohort']
            self.assertEqual(set(delta), {'sample_full', 'sample_holdout'})
            self.assertEqual(delta['sample_full']['test'], dict(final_mAP=1., average_mAP=0.))
            path = extra / 'smoke_evaluation' / 'test' / 'result.json'
            report = json.loads(path.read_text())
            report['stages'][1]['protocol']['reference_scores_sha256'] = 'different-archive'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'common exact reference'):
                summarize(root, smoke=True, full_asl_root=extra)
