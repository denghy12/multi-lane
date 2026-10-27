import copy
import pickle
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.historical_reporting_rng import replay_reporting_rng,verify_historical_boundary
from multi_lane.track_a.task_recovery import rng_state,restore_rng


class HistoricalReportingRNGTest(unittest.TestCase):
    def test_replay_matches_two_real_reporting_iterators(self):
        saved=rng_state()
        try:
            start=torch.get_rng_state()
            for _ in range(2):list(DataLoader(list(range(7)),batch_size=3,shuffle=False,num_workers=0))
            expected=torch.get_rng_state()
            torch.set_rng_state(start)
            record=replay_reporting_rng(0)
            self.assertEqual(record['int64_draws'],2)
            self.assertTrue(torch.equal(torch.get_rng_state(),expected))
            self.assertFalse(record['evaluation_images_read'])
        finally:restore_rng(saved)

    def test_replay_changes_only_cpu_torch_stream(self):
        before=rng_state()
        try:
            replay_reporting_rng(2);after=rng_state()
            self.assertNotEqual(before['torch_cpu'].tolist(),after['torch_cpu'].tolist())
            self.assertEqual(pickle.dumps((before['python'],before['numpy'])),
                             pickle.dumps((after['python'],after['numpy'])))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda'])))
        finally:restore_rng(before)

    def fixture(self,root):
        model=torch.nn.Linear(3,2)
        history=[dict(current_loss=.3,adapter_objective_loss=.1,optimizer_steps=2,skipped_optimizer_steps=0)]*30
        path=root/'recovery_checkpoints/task0_before_reporting.pth';path.parent.mkdir()
        torch.save(dict(model=copy.deepcopy(model.state_dict()),rng=rng_state(),
            provenance=dict(training_history={'0':history})),path)
        return model,history

    def test_reference_comparison_never_installs_parameters_and_is_strict(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,history=self.fixture(root)
            result=verify_historical_boundary(model,history,root,0)
            self.assertTrue(result['all_model_tensors_exact'])
            before=model.weight.clone()
            model.weight.data.add_(1.)
            with self.assertRaises(AssertionError):verify_historical_boundary(model,history,root,0)
            self.assertTrue(torch.equal(model.weight,before+1.))

    def test_reference_detects_loss_and_rng_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model,history=self.fixture(root)
            other=copy.deepcopy(history);other[-1]['current_loss']=.4
            with self.assertRaises(AssertionError):verify_historical_boundary(model,other,root,0)
            saved=rng_state()
            try:
                torch.rand(1)
                with self.assertRaises(AssertionError):verify_historical_boundary(model,history,root,0)
            finally:restore_rng(saved)
