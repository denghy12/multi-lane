import unittest
from collections import Counter

import torch
from torch.utils.data import DataLoader

from scripts.emotic.d9_equal_update_sampler import EqualUpdateSampler


def build_epoch(size, full_size, batch_size, task_seed, epoch):
    generator = torch.Generator().manual_seed(task_seed + epoch)
    sampler = EqualUpdateSampler(size, full_size, batch_size, generator, task_seed, epoch)
    loader = DataLoader(range(size), batch_size=batch_size, sampler=sampler, generator=generator)
    batches = [batch.tolist() for batch in loader]
    return batches, sampler.audit


class EqualUpdateSamplerTest(unittest.TestCase):
    def test_native_shuffle_prefix_and_full_batch_shapes(self):
        size, target, batch_size = 565, 627, 64
        for epoch in (1, 15, 30):
            batches, audit = build_epoch(size, target, batch_size, 60_001, epoch)
            native = DataLoader(range(size), batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(60_001 + epoch))
            prefix = torch.cat(list(native)).tolist()
            rows = [person for batch in batches for person in batch]
            self.assertEqual(rows[:size], prefix)
            self.assertEqual(rows[size:], audit['extra_indices'])
            self.assertEqual(set(rows), set(range(size)))
            self.assertEqual(audit['unique_people'], size)
            self.assertEqual(audit['batch_sizes'], [64]*9 + [51])
            self.assertEqual([len(b) for b in batches], audit['batch_sizes'])
            self.assertEqual(audit['duplicate_within_batches'], 0)
            self.assertEqual(audit, build_epoch(size, target, batch_size, 60_001, epoch)[1])

    def test_supplemental_exposure_balance_at_every_epoch_boundary(self):
        for size, target, batch_size in [(67, 80, 8), (23, 61, 7), (32, 32, 8)]:
            counts = Counter(dict.fromkeys(range(size), 0))
            for epoch in range(1, 31):
                batches, audit = build_epoch(size, target, batch_size, 1, epoch)
                counts.update(audit['extra_indices'])
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                self.assertEqual(sum(counts.values()), epoch * (target - size))
                self.assertEqual(audit['examples'], sum(map(len, batches)))
                self.assertEqual(audit['extra_examples'], target - size)
            with self.assertRaises(ValueError):
                build_epoch(size, size - 1, batch_size, 1, 1)

    def test_actual_eight_task_sizes_have_no_duplicated_batch_members(self):
        for task, (size, target) in enumerate([(4795, 5353), (3955, 4394), (781, 861),
                                                (8936, 9931), (3929, 4352), (2285, 2536),
                                                (565, 627), (1400, 1526)]):
            for epoch in range(1, 31):
                seed = 1 + task * 10_000
                generator = torch.Generator().manual_seed(seed + epoch)
                # The DataLoader reserves the worker seed before drawing the indices.
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                sampler = EqualUpdateSampler(size, target, 64, generator, seed, epoch)
                list(sampler)
                self.assertEqual(sampler.audit['duplicate_within_batches'], 0)
                self.assertEqual(sampler.audit['batch_sizes'],
                                 [min(64, target-i) for i in range(0, target, 64)])
