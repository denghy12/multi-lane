import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.summary_logit_fusion import METHODS, SummaryLogitGate
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.d9_summary_readout import (
    Stage, folds, subset, fit_one, train, evaluate, group_mean_scores, permutation, availability,
)
from scripts.emotic.d9_summary_diagnostics import bootstrap, decision
from scripts.emotic.d9_summary_run import CONFIG, config_check


def population(split='train'):
    n = 12
    ids = np.asarray([f'{split}:image_{i}' for i in range(n)])
    inverse = torch.arange(n).repeat_interleave(2)
    person_ids = np.asarray([f'{name}#person={p}' for name in ids for p in range(2)])
    targets = torch.tensor([[0., 1.], [1., 0.]]).repeat(n, 1)
    sign = targets*2-1
    logits = torch.stack((-2*sign, 4*sign, 2*sign), dim=-1)
    reliable = torch.tensor([True, False, False, False, True, True]).repeat(4)
    descriptor = torch.arange(n*26).float().reshape(n, 26)/100
    return Stage(ids, person_ids, dict(fixed=torch.empty(n, 0), shared=torch.empty(n, 0),
        summary_mean=descriptor[:, :6], summary_correspondence=descriptor), np.full(n, 2),
        reliable.reshape(n, 2).sum(1).numpy(), inverse, reliable, logits, targets,
        dict(native_sha256=split, features_sha256=split, checkpoint_sha256='frozen', config_sha256='config'))


class SummaryPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = dict(steps=4, bias_lr=.01, matrix_lr=.001, amplitude=.25, matrix_l2=.001,
                           folds=3, permutations=2, permutation_seed=17, bootstrap=4, bootstrap_seed=19)

    def test_all_train_adds_four_controls_without_overrides(self):
        cfg = json.loads(CONFIG.read_text())
        config_check(cfg)
        self.assertEqual(len(cfg['sources'])*len(cfg['methods']), 8)
        cfg['sources']['full']['steps'] = 200
        with self.assertRaises(ValueError):
            config_check(cfg)

    def test_original_image_folds_and_no_shared_people(self):
        stage = population()
        assignment = folds(stage.ids)
        self.assertEqual(set(assignment.tolist()), {0, 1, 2})
        for fold in range(3):
            fitting, held = subset(stage, np.flatnonzero(assignment != fold)), subset(stage, np.flatnonzero(assignment == fold))
            self.assertFalse(set(fitting.ids) & set(held.ids))
            self.assertFalse(set(fitting.person_ids) & set(held.person_ids))
            np.testing.assert_array_equal(folds(held.ids), np.full(len(held.ids), fold))
            self.assertTrue(torch.bincount(held.inverse).eq(2).all())

    def test_fit_supervision_and_train_only_center(self):
        stage = population()
        result = fit_one(stage, 'summary_mean', self.config)
        self.assertLess(result['bce_history'][-1], result['bce_history'][0])
        gate = SummaryLogitGate.restore(result['gate'])
        torch.testing.assert_close(gate.center, stage.descriptors['summary_mean'].mean(0))
        center = gate.center.clone()
        gate(stage.descriptors['summary_mean']+100)
        torch.testing.assert_close(gate.center, center, atol=0, rtol=0)

    def test_exchange_preserves_availability_and_mean_lock_reuses_val(self):
        stage = population('val')
        scores = torch.randn(len(stage.ids), 3)
        order = permutation(stage, np.random.default_rng(1))
        np.testing.assert_array_equal(availability(stage)[order], availability(stage))
        self.assertTrue((order != torch.arange(len(order))).all())
        original, lock = group_mean_scores(scores, stage)
        reused, second = group_mean_scores(scores+100, stage, lock)
        torch.testing.assert_close(original, reused, atol=0, rtol=0)
        self.assertEqual(lock, second)

    def test_train_val_seal_test_and_eight_row_ingredients(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            def load(source, features, split, task, smoke):
                return population(split)
            with patch('scripts.emotic.d9_summary_readout.load_stage', side_effect=load):
                train(out, out, out, [0, 6], self.config, True)
                with self.assertRaises(FileNotFoundError):
                    evaluate(out, out, out, 'test', [0, 6], self.config, True)
                evaluate(out, out, out, 'val', [0, 6], self.config, True)
                interval = bootstrap(out, 'val', 4, 19)
                self.assertEqual(interval['valid'], 4)
                verdict = decision(out)
                self.assertFalse(any(v['eligible_for_separate_eight_task_plan'] for v in verdict.values()))
                test = evaluate(out, out, out, 'test', [0, 6], self.config, True)
                self.assertEqual(set(test['mean_current_head_mAP']), set(METHODS))
                for row in read(out/'fit.json')['stages']:
                    for m in METHODS:
                        self.assertEqual(len(row['methods'][m]['folds']), 3)
                # A changed TRAIN state cannot slip through the validation seal.
                (out/'fit.json').write_text((out/'fit.json').read_text()+'\n')
                with self.assertRaises(ValueError):
                    evaluate(out, out, out, 'test', [0, 6], self.config, True)


if __name__ == '__main__':
    unittest.main()
