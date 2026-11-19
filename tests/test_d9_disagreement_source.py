import unittest

import numpy as np

from scripts.emotic.d9_disagreement_source import _check_train


class FullTrainCohortTest(unittest.TestCase):
    def test_requires_full_current_cohort(self):
        data = dict(targets=np.array([[1, 0], [1, 1]]), reliable=np.array([True, False]),
                    sample_ids=np.array(['train:a#person=0', 'train:a#person=1']))
        record = dict(people=2, images=1, classes=[4, 5], sample_id_sha256='digest',
                      reliable_face=1, positives=[2, 1], reliable_face_positives=[1, 0])
        cohort = dict(branch=record, fusion=record)
        _check_train(data, (4, 5), 'digest', cohort, {'train:a': 'both'})
        with self.assertRaisesRegex(ValueError, 'cohort'):
            _check_train(data, (4, 5), 'digest', cohort, {'train:a': 'fusion'})
        with self.assertRaisesRegex(ValueError, 'cohort'):
            _check_train(data, (5, 4), 'digest', cohort, {'train:a': 'both'})
