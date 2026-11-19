import unittest
from types import SimpleNamespace

from multi_lane.track_a.runner import task_indices
from scripts.emotic.d9_matched_cohort import pair_within_reliability, select_groups
from scripts.emotic.d9_sample_training import describe_cohort


class MatchedCohortTest(unittest.TestCase):
    def setUp(self):
        self.s = SimpleNamespace(sample_ids=[f'train:im{i}#person=0' for i in range(7)],
            targets=[[0], [1], [0], [0,1], [0], [1], [0,1]], face_reliable=[1,0,1,1,1,0,1],
            face_training_valid=[1,0,1,1,1,0,1])
        self.roles = {f'train:im{i}':'branch' if i < 4 else 'fusion' for i in range(7)}
        classes = task_indices(0)
        self.arch = {k:describe_cohort(self.s, idx, classes)
                     for k,idx in [('branch', list(range(4))), ('fusion', [4,5,6])]}

    def test_exact_matching_native_order_and_disjoint_images(self):
        rows, audit = select_groups(self.s, 0, self.roles, self.arch)
        self.assertEqual(rows['held'], [4,5,6])
        self.assertEqual(len(set(rows['seen'])), 3)
        self.assertEqual((audit['exact_label_vectors'],audit['total_label_mismatches']), (3,0))
        self.assertEqual(audit['seen']['by_reliability'], audit['held']['by_reliability'])
        self.assertEqual(audit['original_image_overlap'], 0)
        self.assertEqual((rows,audit),select_groups(self.s, 0, self.roles, self.arch))

    def test_prefers_matching_face_training_eligibility_and_minimum_hamming(self):
        # Row 4 wants [0] with valid Face training. Row 0 has the right label but is invalid.
        self.s.face_training_valid[0] = 0
        self.s.targets[2] = [0,1]
        self.s.targets[3] = [0,1,2]
        self.assertEqual(pair_within_reliability(self.s, [0,2,3], [4], (0,1,2), 0, 1), [2])
        # Only the invalid candidate remains; preserve the held person and report this fallback.
        self.assertEqual(pair_within_reliability(self.s, [0], [4], (0,1,2), 0, 1), [0])

    def test_reject_invalid_source_or_manifests(self):
        with self.assertRaisesRegex(ValueError,'registered'):
            select_groups(self.s, 0, self.roles, {'branch':{}, 'fusion':{}})
        with self.assertRaisesRegex(ValueError,'Too few'):
            pair_within_reliability(self.s, [], [4], (0,), 0, 1)
