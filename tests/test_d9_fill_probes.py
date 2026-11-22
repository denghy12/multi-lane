"""Protect the data role distinction in the equal-update missing-table supplement."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.emotic.d9_fill_probes import check_train_population, require_source
from scripts.emotic.d9_sample_training import image_key


class TestHeldPopulation(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(['train:imageA.jpg#person=0', 'train:imageA.jpg#person=1',
                             'train:imageB.jpg#person=0'])
        self.data = dict(sample_ids=self.ids, targets=np.array([[1, 0], [0, 1], [1, 1]]),
                         reliable=np.array([True, False, True]))
        images = {image_key(i) for i in self.ids}
        self.roles = {k: 'fusion' for k in images}
        self.digest = hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()
        self.cohort = {'fusion': dict(people=3, images=len(images), classes=[0, 1],
            sample_id_sha256=self.digest, reliable_face=2, positives=[2, 2],
            reliable_face_positives=[2, 1])}

    def test_accepts_exact_held_cohort(self):
        self.assertEqual(check_train_population(self.data, self.cohort, self.roles, [0, 1]), self.digest)

    def test_rejects_branch_seen_people_even_when_counts_match(self):
        self.roles[image_key(self.ids[0])] = 'branch'
        with self.assertRaisesRegex(ValueError, 'permanent held'):
            check_train_population(self.data, self.cohort, self.roles, [0, 1])

    def test_rejects_changed_label_coverage(self):
        self.data['targets'][0, 1] = 1
        with self.assertRaisesRegex(ValueError, 'permanent held'):
            check_train_population(self.data, self.cohort, self.roles, [0, 1])

    def test_rejects_permuted_person_order(self):
        shuffled = {k: v[[2, 1, 0]] for k, v in self.data.items()}
        with self.assertRaisesRegex(ValueError, 'permanent held'):
            check_train_population(shuffled, self.cohort, self.roles, [0, 1])


class TestEqualUpdateSource(unittest.TestCase):
    def test_rejects_original_shorter90_source(self):
        cfg = dict(mode='sample_holdout', seed=1, smoke=False, tasks=8,
                   epochs=30, batch_size=64, protect_branches=True,
                   image_split='global_10pct_hash_rank')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'equal-update'):
                require_source(root)
            cfg['equal_updates'] = {'mode': 'cycle_90pct_to_full_epoch_size'}
            (root / 'config.json').write_text(json.dumps(cfg))
            self.assertEqual(require_source(root), cfg)


if __name__ == '__main__':
    unittest.main()
