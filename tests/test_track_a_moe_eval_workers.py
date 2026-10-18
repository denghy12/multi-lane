import os
import unittest
import torch
from torch.utils.data import Dataset
from multi_lane.track_a.evaluate_moe_task_ffn import evaluation_loader


class ProcessDataset(Dataset):
    def __len__(self):return 5
    def __getitem__(self,index):return index,os.getpid()


class EvaluationWorkersTest(unittest.TestCase):
    def test_zero_override_runs_in_parent_and_preserves_order_and_partial_batch(self):
        config={'workers':2}
        direct=evaluation_loader(ProcessDataset(),3,config,workers=0)
        default=evaluation_loader(ProcessDataset(),3,config)
        left=list(direct);right=list(default)
        self.assertEqual([len(x[0]) for x in left],[3,2])
        self.assertTrue(torch.equal(torch.cat([x[0] for x in left]),torch.cat([x[0] for x in right])))
        self.assertTrue(all(bool((x[1]==os.getpid()).all()) for x in left))
        self.assertTrue(all(bool((x[1]!=os.getpid()).all()) for x in right))
        self.assertEqual(config,{'workers':2})

    def test_invalid_worker_override_rejected(self):
        for workers in (-1,.5,True):
            with self.subTest(workers=workers),self.assertRaises(ValueError):
                evaluation_loader(ProcessDataset(),3,{'workers':2},workers)


if __name__=='__main__':unittest.main()
