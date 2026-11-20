import unittest

import numpy as np

from scripts.emotic.d9_same_image_neg_fit import join_population


def sample(pid, positive, reliable):
    y = np.array([[positive, False]])
    z = np.zeros((1, 2, 3), dtype=np.float16)
    return dict(sample_ids=np.array([pid]), targets=y, reliable=np.array([reliable]),
                branch_logits=z, person_branch_logits=z.copy(), face_branch_logits=z.copy())


class SameImagePopulationTest(unittest.TestCase):
    def setUp(self):
        self.original = sample('train:image-a#person=1', True, True)
        self.id = 'train:image-a#person=2'
        self.added = sample(self.id, False, False)
        self.prior = {self.id: False, 'train:image-b#person=3': True}

    def test_adds_only_eligible_previous_people_once(self):
        joined = join_population(self.original, self.added, self.prior)
        self.assertEqual(joined['targets'].tolist(), [[True, False], [False, False]])
        self.assertEqual(joined['reliable'].tolist(), [True, False])
        wrong_image = sample('train:image-b#person=3', False, True)
        with self.assertRaisesRegex(ValueError, 'roster'):
            join_population(self.original, wrong_image, self.prior)
        duplicated = {k: np.concatenate([v, v]) for k,v in self.added.items()}
        with self.assertRaisesRegex(ValueError, 'roster'):
            join_population(self.original, duplicated, self.prior)

    def test_requires_confirmed_negative_label_and_same_face_quality(self):
        self.added['targets'][0, 1] = True
        with self.assertRaisesRegex(ValueError, 'all-negative'):
            join_population(self.original, self.added, self.prior)
        self.added['targets'][0, 1] = False
        self.added['reliable'][0] = True
        with self.assertRaisesRegex(ValueError, 'availability'):
            join_population(self.original, self.added, self.prior)
