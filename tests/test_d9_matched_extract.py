import unittest
from types import SimpleNamespace

import numpy as np
from scripts.emotic.d9_matched_extract import verify_design, verify_intervention


class MatchedExtractionTest(unittest.TestCase):
    def test_original_image_roles_and_unique_people(self):
        source = SimpleNamespace(sample_ids=['train:a#person=0', 'train:b#person=0', 'train:a#person=1'])
        roles = {'train:a': 'branch', 'train:b': 'fusion'}
        verify_design(source, {'held':[1], 'seen':[0]}, roles)
        with self.assertRaises(ValueError):
            verify_design(source, {'held':[2], 'seen':[0]}, roles)
        with self.assertRaises(ValueError):
            verify_design(source, {'held':[1,1], 'seen':[0,2]}, roles)

    def test_only_eligible_face_rows_change_and_full_stays_unchanged(self):
        r = np.array([True, False])
        natural = dict(branch_logits=np.zeros((2,3,3)), content=np.zeros((2,3,12)), relation=np.zeros((2,8)))
        variant = {k:v.copy() for k,v in natural.items()}
        variant['branch_logits'][0,0,1] = 1
        verify_intervention(natural, variant, 'face', r)
        variant['relation'][1,0] = 1
        with self.assertRaises(AssertionError):
            verify_intervention(natural, variant, 'face', r)
        variant['branch_logits'][0,0,0] = 1
        with self.assertRaises(AssertionError):
            verify_intervention(natural, variant, 'person', r)
