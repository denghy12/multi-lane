"""Expose only branch-role people while matching the full-TRAIN epoch length."""
import hashlib

import numpy as np
import torch
from torch.utils.data import RandomSampler, Sampler


class EqualUpdateSampler(Sampler):
    """Yield positions in the branch dataset, never positions in its held-out complement.

    Pass the same generator to this sampler and the DataLoader. The loader consumes its
    usual worker seed before drawing the first index. RandomSampler then reproduces the
    unmodified shuffle=True prefix. Supplemental draws use independent generators.
    """

    def __init__(self, size, full_size, batch_size, generator, task_seed, epoch):
        if size < 1 or full_size < size or batch_size < 1 or epoch < 1:
            raise ValueError("Invalid branch size, target epoch size, batch size or epoch")
        self.size = size
        self.full_size = full_size
        self.batch_size = batch_size
        self.generator = generator
        self.task_seed = task_seed
        self.epoch = epoch
        self.audit = None

    def __len__(self):
        return self.full_size

    def __iter__(self):
        order = list(RandomSampler(range(self.size), generator=self.generator))
        extra_count = self.full_size - self.size
        rotation = torch.randperm(self.size, generator=torch.Generator().manual_seed(
            7_000_001 + self.task_seed)).tolist()
        shuffle = torch.Generator().manual_seed(8_000_001 + self.task_seed + self.epoch)
        start = (self.epoch - 1) * extra_count

        # Every prefix of completed epochs draws each person equally often, within one.
        # Split into unique-person passes in case the requested extension exceeds one pass.
        for offset in range(0, extra_count, self.size):
            count = min(self.size, extra_count - offset)
            chosen = [rotation[(start + offset + j) % self.size]
                      for j in torch.randperm(count, generator=shuffle).tolist()]
            partial = len(order) % self.batch_size
            if partial:
                previous = set(order[-partial:])
                take = self.batch_size - partial
                safe = [person for person in chosen if person not in previous][:take]
                safe_set = set(safe)
                chosen = safe + [person for person in chosen if person not in safe_set]
            order.extend(chosen)

        batches = [order[i:i+self.batch_size] for i in range(0, len(order), self.batch_size)]
        self.audit = dict(unique_people=len(set(order)), examples=len(order),
                          extra_examples=extra_count, extra_indices=order[self.size:],
                          batch_sizes=[len(batch) for batch in batches],
                          duplicate_within_batches=sum(len(batch) - len(set(batch)) for batch in batches),
                          sequence_sha256=hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest())
        yield from order
