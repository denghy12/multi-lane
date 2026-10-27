import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Dataset

from multi_lane.track_a.self_trained_fast_evaluation import (
    code_audit, input_digest, loader_for, validate_saved)
from multi_lane.track_a.runner import compute_metrics
from multi_lane.track_a.run_historical_b32_test_attribution import write


class FixedInputs(Dataset):
    def __len__(self): return 70

    def __getitem__(self, index):
        x=torch.tensor([index, index+1.], dtype=torch.float32)
        return dict(full=x,person=x+1,face=x+2,face_reliable=torch.tensor(bool(index%2))), \
               torch.tensor([float(index%2)]), 'person_'+str(index)


class FastEvaluationTest(unittest.TestCase):
    def test_workers_preserve_pixels_labels_masks_ids_and_partial_batch(self):
        old=loader_for(FixedInputs(),0,0)
        new=loader_for(FixedInputs(),0,2)
        a=[input_digest(b) for b in old]
        b=[input_digest(b) for b in new]
        self.assertEqual(a,b)
        self.assertEqual(b,[input_digest(b) for b in new])
        self.assertEqual(new.prefetch_factor,1)
        self.assertTrue(new.persistent_workers)
        self.assertEqual(new.batch_size,64)

    def fixture(self, root):
        source=SimpleNamespace(sample_ids=['a','b'], targets=[[0,3],[1]],face_reliable=[True,False])
        view=SimpleNamespace(source=source,indices=[0,1],class_indices=[0,1])
        targets=np.eye(2,dtype=np.float32)
        probabilities=np.asarray([[.8,.2],[.3,.7]],dtype=np.float32)
        metrics=asdict(compute_metrics(0,torch.from_numpy(probabilities),torch.from_numpy(targets),.5))
        values=dict(task_id=np.asarray(0),sample_ids=np.asarray(['a','b']),targets=targets,
                    class_indices=np.arange(2),batch_lengths=np.asarray([2]),
                    face_reliable=np.asarray([True,False]))
        for name in ('fused','full','person','face'):
            values[name+'_probabilities']=probabilities
            values[name+'_logits']=probabilities
        (root/'view_scores').mkdir();(root/'scores').mkdir()
        np.savez_compressed(root/'view_scores/task0.npz',**values)
        np.savez_compressed(root/'scores/task0.npz',**values,probabilities=probabilities,logits=probabilities)
        write(root/'task0.json',dict(metrics=metrics,views=dict(metrics={
            name:metrics for name in ('fused','full','person','face')})))
        return view

    def test_only_complete_valid_triplets_are_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);view=self.fixture(root)
            result=validate_saved(root,0,view)
            self.assertEqual(len(result['files']),3)
            (root/'task0.json').unlink()
            self.assertIsNone(validate_saved(root,0,view))

    def test_reuse_rejects_wrong_people_masks_metrics_and_class_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);view=self.fixture(root)
            with patch.object(view.source,'sample_ids',['b','a']):
                with self.assertRaises(AssertionError):validate_saved(root,0,view)
            with patch.object(view.source,'face_reliable',[False,False]):
                with self.assertRaises(AssertionError):validate_saved(root,0,view)
            with patch.object(view,'class_indices',[1,0]):
                with self.assertRaises(AssertionError):validate_saved(root,0,view)
            write(root/'task0.json',dict(metrics={'mAP':0},views={'metrics':{'fused':{}}}))
            with self.assertRaises(AssertionError):validate_saved(root,0,view)

    def test_existing_package_code_must_remain_exact(self):
        with patch('subprocess.check_output',return_value='multi_lane/track_a/self_trained_fast_evaluation.py\n'):
            self.assertTrue(code_audit('source')['existing_package_files_byte_identical'])
        with patch('subprocess.check_output',return_value='multi_lane/track_a/model.py\n'):
            with self.assertRaises(ValueError):code_audit('source')
