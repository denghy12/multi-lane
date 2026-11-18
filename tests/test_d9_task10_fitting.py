import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.emotic.fit_d9_task10_probe import (
    TASK_SIZES, build_features, combine_message, load_arrays, run, selection
)


def make_cache(root, split, seed):
    directory = root / 'features/task0'
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n, c, width = 18, TASK_SIZES[0], 16
    reliable = np.arange(n) % 3 != 0
    labels = rng.integers(0, 2, (n, c))
    z = rng.standard_normal((n, c, 3)).astype('float32')
    content = rng.standard_normal((n, 3, width)).astype('float32')
    delta = .01 * rng.standard_normal((n, 2, width)).astype('float32')
    content[~reliable, 2] = np.nan
    delta[~reliable, 1] = np.nan
    path = directory / f'{split}.npz'
    np.savez(path, sample_ids=[f'{split}-{i}' for i in range(n)], targets=labels,
             reliable=reliable, branch_logits=z, relation=rng.random((n,8)),
             content=content, delta=delta)
    return path


class Task10ProbeFitting(unittest.TestCase):
    def test_projection_paired_capacity_masks_and_donor_isolation(self):
        with TemporaryDirectory() as tmp:
            d = load_arrays(make_cache(Path(tmp),'train',0),0)
            f, delta, stat = build_features(d)
            self.assertEqual(f['C'].shape, f['D'].shape)
            np.testing.assert_array_equal(f['D'], combine_message(f['C'],delta))
            self.assertTrue(np.isfinite(f['D']).all())
            np.testing.assert_array_equal(f['D'][~d['reliable'],8:],0.)
            retained = [0,1,2,3,4,5,8,9]
            np.testing.assert_array_equal(f['C'][:,retained], f['D'][:,retained])
            np.testing.assert_allclose(np.asarray(stat['message']['mean']).shape,(2,4))

    def test_validation_lock_test_phase_and_tamper_detection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, split in enumerate(('train','val')):
                make_cache(root,split,seed)
            run(root,'fit-val',[0],4,3)
            lock = json.loads((root/'probe/validation_lock.json').read_text())
            self.assertEqual(lock['selection_split'],'val')
            self.assertEqual(lock['protocol']['steps'],4)
            self.assertFalse((root/'features/task0/test.npz').exists())
            make_cache(root,'test',2)
            run(root,'test',[0],4,3)
            model = json.loads((root/'probe/task0_model.json').read_text())
            self.assertEqual(np.shape(model['matrices']['C']),np.shape(model['matrices']['D']))
            report = json.loads((root/'probe/task0_test.json').read_text())
            self.assertEqual(report['fixed']['actual_minus_group_mean'],0.)
            self.assertEqual(len(report['D']['message_shuffle']['differences']),3)
            self.assertEqual(len(report['D']['metrics']['all']['per_class_ap']),TASK_SIZES[0])
            (root/'probe/task0_model.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'lock'):
                run(root,'test',[0],4,3)
