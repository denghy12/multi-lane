import unittest
from types import SimpleNamespace

import torch

from scripts.emotic.d9_sample_training import (
    describe_cohort, epoch_groups, fusion_source, global_image_roles, image_key, smoke_rows, task_cohorts)


class TestSampleTraining(unittest.TestCase):
    @staticmethod
    def source():
        ids = [f'train:folder/{img}.jpg#person={person}' for img in range(100) for person in (0, 1, 2)]
        return SimpleNamespace(sample_ids=ids, targets=[[n % 4, 4] for n in range(len(ids))],
                               face_reliable=[n % 3 == 0 for n in range(len(ids))],
                               face_training_valid=[n % 3 != 2 for n in range(len(ids))])

    def test_original_image_roles_persist_across_tasks_and_row_order(self):
        source = self.source()
        roles = global_image_roles(source, 'sample_holdout', 1)
        self.assertEqual(sum(role == 'fusion' for role in roles.values()), 10)
        self.assertEqual(roles, global_image_roles(SimpleNamespace(sample_ids=source.sample_ids[::-1]),
                                                   'sample_holdout', 1))
        for label in (0, 2, 4):
            eligible = [i for i, target in enumerate(source.targets) if label in target]
            branch, fusion = task_cohorts(source, eligible, roles, False)
            branch_images = {image_key(source.sample_ids[i]) for i in branch}
            gate_images = {image_key(source.sample_ids[i]) for i in fusion}
            self.assertFalse(branch_images & gate_images)
            self.assertEqual(sorted(branch + fusion), eligible)
            self.assertEqual({roles[key] for key in gate_images}, {'fusion'})
        self.assertEqual(describe_cohort(source, fusion, [4])['positives'], [len(fusion)])

    def test_full_uses_all_rows_and_matches_legacy_smoke(self):
        source = self.source()
        roles = global_image_roles(source, 'sample_full', 1)
        self.assertEqual(roles, global_image_roles(source, 'sample_full_asl', 1))
        ids = list(range(len(source.sample_ids)))
        self.assertEqual(task_cohorts(source, ids, roles, False), (ids, ids))
        sample = smoke_rows(source, ids)
        self.assertEqual(sample, [1, 2, 4, 5, 0, 3, 6, 9])
        self.assertEqual(task_cohorts(source, ids, roles, True), (sample, sample))

    def test_groups_are_complete_balanced_and_use_private_rng(self):
        state = torch.get_rng_state()
        groups = epoch_groups(117, 10, 7, torch.Generator().manual_seed(124))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        updates = [[i for micro in step for i in micro] for step in groups]
        self.assertEqual(len(updates), 10)
        self.assertEqual(sorted(i for step in updates for i in step), list(range(117)))
        self.assertLessEqual(max(map(len, updates)) - min(map(len, updates)), 1)
        self.assertLessEqual(max(len(micro) for step in groups for micro in step), 7)
        with self.assertRaises(ValueError):
            epoch_groups(3, 4, 2, torch.Generator())

    def test_deterministic_gate_inputs_do_not_change_branch_transforms(self):
        src = SimpleNamespace(train=True, transforms_by_view={'full': ('train', 'eval')})
        det = fusion_source(src)
        self.assertTrue(src.train)
        self.assertEqual(src.transforms_by_view['full'], ('train', 'eval'))
        self.assertFalse(det.train)
        self.assertEqual(det.transforms_by_view['full'], ('eval', 'eval'))


if __name__ == '__main__':
    unittest.main()
