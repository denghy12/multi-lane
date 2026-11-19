import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.emotic.d9_shrink_eval import aggregate, blend, evaluate_task, run
from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.fit_d9_task10_probe import apply_rule, build_features, combine_message, load_arrays, mix, scores
from scripts.emotic.d9_direction_permutation import permutation_sources
from test_d9_task10_fitting import make_cache
from scripts.emotic.fit_d9_task10_probe import run as run_previous


class ShrinkEvaluationTests(unittest.TestCase):
    def test_endpoints_and_content_preserving_message_exchange(self):
        with TemporaryDirectory() as tmp:
            d = load_arrays(make_cache(Path(tmp), 'val', 144), 0)
            features, msg, stats = build_features(d)
            rng = np.random.default_rng(312)
            model = dict(feature_stats=stats, center=[[.3, -.1, 0.], [.2, .1, -.1]],
                         matrices={'D': rng.normal(size=(3,12)).tolist()})
            seeds = (551, 552, 553)
            row0 = evaluate_task(d, model, 0, 0., seeds)
            self.assertEqual(row0['metrics']['blend'], row0['metrics']['A'])
            self.assertEqual(row0['minus_group_mean'], 0.)
            for control in ('message_shuffle', 'weight_shuffle'):
                self.assertEqual(row0[control]['differences'], [0.]*len(seeds))
            self.assertFalse(aggregate([row0])['passes_criteria'])

            row1 = evaluate_task(d, model, 0, 1., seeds)
            self.assertEqual(row1['metrics']['blend'], row1['metrics']['D'])
            val_means = [[.6,.4,0.], [.5,.3,.2]]
            row = evaluate_task(d, model, 0, .37, seeds, val_means)
            donor = permutation_sources(d['sample_ids'], d['reliable'], 1, seeds[0])[:,0]
            swapped = combine_message(features['C'], msg[donor])
            a = apply_rule('A', d, features, model)
            wrong_d = apply_rule('D', d, features, model, swapped)
            z = torch.from_numpy(d['branch_logits']).float()
            expected = scores(mix(.63*a+.37*wrong_d, z), d['targets'])['mAP']
            self.assertAlmostEqual(row['message_shuffle']['differences'][0],
                                   row['metrics']['blend']['all']['mAP'] - expected)
            mean_weights = torch.tensor(val_means)[d['reliable'].astype(int)]
            expected_mean = scores(mix(mean_weights,z),d['targets'])['mAP']
            self.assertEqual(row['metrics']['group_mean']['all']['mAP'], expected_mean)
            self.assertEqual(row['actual_group_weights'][0][2], 0.)

    def test_test_phase_requires_untampered_validation_seal(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'previous'
            for seed, split in enumerate(('train','val','test')):
                make_cache(source, split, seed)
            run_previous(source, 'fit-val', [0], 3, 2)
            run_previous(source, 'test', [0], 3, 2)
            out = Path(tmp)/'shrink'
            out.mkdir()
            lock = {'lambda': .37, 'tasks': [0],
                    'source_hashes': {'0': sha256_file(source/'features/task0/train.npz')}}
            (out/'lambda_lock.json').write_text(json.dumps(lock))
            with patch('scripts.emotic.d9_shrink_fit.read_lock', return_value=lock):
                with self.assertRaises(FileNotFoundError):
                    run(source, out, 'test', 2)
                run(source, out, 'val', 2)
                run(source, out, 'test', 2)
                report = json.loads((out/'test_results.json').read_text())
                self.assertEqual(report['protocol']['fraction'], .37)
                self.assertTrue(report['endpoint_replay'])
                (out/'val_results.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'SHA-256'):
                    run(source, out, 'test', 2)

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            blend(torch.zeros(1,3), torch.zeros(1,3), float('nan'))
