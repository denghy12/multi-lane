import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.emotic.summarize_d9_refinement import (
    ANCHOR_VIEWS, add_metrics, branch_paths, finish_metrics, fusion_groups, pair_ranking,
)


class RefinementSummaryTests(unittest.TestCase):
    def test_pairs_compare_same_image_and_opposite_class_only(self):
        ids = ['val:/a/img#person=1', 'val:/a/img#person=0', 'val:/b/img#person=0',
               'val:/a/img#person=2']
        labels = np.array([[1, 1], [0, 0], [0, 1], [1, 0]], dtype=np.float32)
        logits = np.array([[5, 0], [0, 0], [9, 9], [-1, 1]], dtype=np.float32)
        row = pair_ranking(ids, logits, labels)
        self.assertEqual((row['multi_images'], row['multi_people'], row['person_pairs']), (1, 3, 3))
        self.assertEqual(row['different_label_pairs'], 3)
        self.assertEqual(row['pooled'], dict(comparisons=4, correct=1, ties=1,
                                              incorrect=2, accuracy_half_credit=.375))
        bad_ids = ['x', *ids[1:]]
        with self.assertRaisesRegex(ValueError, 'person ID'):
            pair_ranking(bad_ids, logits, labels)

    def test_pair_ties_give_half_credit_and_singletons_no_denominator(self):
        ids = ['val:x/a#person=0', 'val:x/a#person=1']
        logits = np.zeros((2, 2), dtype=np.float32)
        labels = np.eye(2, dtype=np.float32)
        tied = pair_ranking(ids, logits, labels)
        self.assertEqual(tied['identical_logit_vector_pairs'], 1)
        self.assertEqual(tied['pooled']['accuracy_half_credit'], .5)
        single = pair_ranking(ids[:1], logits[:1], labels[:1])
        self.assertIsNone(single['pooled']['accuracy_half_credit'])

    def test_registered_readouts_match_requested_branch_sources(self):
        paths = branch_paths(Path('new'), Path('anchor'), False)
        groups = fusion_groups(False)
        self.assertEqual(groups['D'], ANCHOR_VIEWS)
        self.assertEqual(len(groups), 10)
        self.assertEqual(set(paths), set(k for items in groups.values() for k in items))
        self.assertEqual(groups['sync_person'], ('full', 'sync_person', 'face_all'))
        self.assertEqual(groups['independent_both'], ('full', 'independent_person', 'independent_face'))
        self.assertEqual(groups['cap035_person'], ('full', 'cap035_person', 'face_all'))
        self.assertEqual(groups['box_swap_full'], ('box_swap_full', 'person_all', 'face_all'))
        self.assertEqual(paths['box_swap_full'], Path('new/box_swap/native/full'))
        self.assertEqual(set(fusion_groups(True)), {'D'})

    def test_incomplete_history_is_never_an_eight_stage_average(self):
        rows = {}
        logits = torch.tensor([[1.], [-1.]])
        labels = torch.tensor([[1.], [0.]])
        add_metrics(rows, 'full', 0, logits, labels, {'all': torch.tensor([True, True])})
        self.assertAlmostEqual(rows['full']['all'][0].mAP, 100., places=5)
        with self.assertRaisesRegex(ValueError, 'Incomplete phase'):
            finish_metrics(rows)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            add_metrics(rows, 'full', 0, logits, labels, {'reliable': torch.tensor([False, False])})


if __name__ == '__main__':
    unittest.main()
