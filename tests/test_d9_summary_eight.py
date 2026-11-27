from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.runner import TASK_SIZES
from multi_lane.track_a.summary_logit_fusion import METHODS
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.d9_summary_eight import (
    CONFIG, FIT_PROTOCOL, BENCHMARK, config_check, aggregate, evaluate,
)
from scripts.emotic.d9_summary_readout import train
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from tests.test_d9_summary_pipeline import population


def stage_population(split, task, current_only=True):
    stage = population(split)
    classes = TASK_SIZES[task] if split == 'train' or current_only else sum(TASK_SIZES[:task+1])
    targets = torch.arange(24*classes).reshape(24, classes).remainder(3).eq(0).float()
    generator = torch.Generator().manual_seed(123+task)
    logits = torch.randn(24, classes, 3, generator=generator)
    return replace(stage, targets=targets, logits=logits)


class SummaryEightTests(unittest.TestCase):
    def test_locked_original_benchmark_and_full_stage_configuration(self):
        cfg = read(CONFIG)
        config_check(cfg)
        self.assertEqual(BENCHMARK['test'], dict(final_mAP=33.1770, average_mAP=40.2083))
        cfg['tasks'] = [0, 3, 6]
        with self.assertRaises(ValueError):
            config_check(cfg)

    def test_aggregate_rejects_partial_and_current_head_results(self):
        with self.assertRaises(ValueError):
            aggregate([], False)
        rows = []
        for task in range(8):
            stage = stage_population('val', task, False)
            metrics = metrics_for_stage(task, stage.targets, stage.reliable,
                                        {m: stage.logits[..., 0] for m in METHODS})
            rows.append(dict(task=task, metrics=metrics))
        result = aggregate(rows, False)
        for m in METHODS:
            self.assertEqual(result[m]['all']['final_mAP'], rows[-1]['metrics'][m]['all']['mAP'])
            self.assertAlmostEqual(result[m]['all']['average_mAP'],
                                   np.mean([r['metrics'][m]['all']['mAP'] for r in rows]))
        rows[-1]['metrics']['shared']['all']['seen_classes'] = TASK_SIZES[-1]
        with self.assertRaises(ValueError):
            aggregate(rows, False)

    def test_eight_stage_pipeline_uses_current_gate_for_all_old_and_new_classes(self):
        config = dict(steps=4, bias_lr=.01, matrix_lr=.001, amplitude=.25, matrix_l2=.001,
                      folds=3, permutations=2, permutation_seed=17, bootstrap=4, bootstrap_seed=19)
        def load(source, features, split, task, smoke, *, current_only=True):
            return stage_population(split, task, current_only)
        def fixed(source, features, split, task, stage):
            return fuse_logits(*stage.logits.unbind(-1), stage.reliable)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root/'source', root/'readout'
            for split in ('val', 'test'):
                rows = []
                for task in range(8):
                    stage = stage_population(split, task, False)
                    metrics = metrics_for_stage(task, stage.targets, stage.reliable,
                                                {m: fixed(source, root, split, task, stage) for m in METHODS})
                    rows.append(dict(task=task, scores_sha256=split, metrics=metrics))
                path = source/'evaluation'/split/'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(smoke=False, split=split, stages=rows, summary=aggregate(rows, False))))
            with patch('scripts.emotic.d9_summary_readout.load_stage', side_effect=load), \
                    patch('scripts.emotic.d9_summary_eight.load_stage', side_effect=load), \
                    patch('scripts.emotic.d9_summary_eight.canonical_fixed', side_effect=fixed):
                train(source, root, out, list(range(8)), config, False, protocol=FIT_PROTOCOL)
                with self.assertRaises(FileNotFoundError):
                    evaluate(source, root, out, 'test', list(range(8)), config, False)
                evaluate(source, root, out, 'val', list(range(8)), config, False)
                result = evaluate(source, root, out, 'test', list(range(8)), config, False)
                self.assertTrue(result['target_comparable'])
                self.assertEqual(len(result['stages']), 8)
                with np.load(out/'test'/'task7.npz', allow_pickle=False) as z:
                    self.assertEqual(z['targets'].shape[1], 26)
                    stage = stage_population('test', 7, False)
                    for m in METHODS[1:]:
                        expected = (stage.logits.numpy()*z[m+'_weights'][:, None, :]).sum(2)
                        np.testing.assert_array_equal(z[m+'_logits'], expected)
                        self.assertTrue(np.all(z[m+'_weights'][~z['reliable'], 2] == 0))
                (out/'fit.json').write_text((out/'fit.json').read_text()+'\n')
                with self.assertRaises(ValueError):
                    evaluate(source, root, out, 'test', list(range(8)), config, False)


if __name__ == '__main__':
    unittest.main()
