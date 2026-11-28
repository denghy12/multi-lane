import unittest
import torch
from test_d9_final_patch import population,args
from multi_lane.track_a.task_patch_fusion import TaskPatchFusion,candidates
from multi_lane.track_a.task_patch_execution import PreparedDescriptor
from scripts.emotic.d9_task_patch_readout import fit_one
from scripts.emotic.d9_task_patch_run import FIT


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False

    def test_prepared_descriptor_exact_values_and_gradients(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            stage=population();inputs=tuple(x.to(device) for x in args(stage))
            spec=next(s for s in candidates() if s['key']=='cls_patch__common')
            gate=TaskPatchFusion(spec).to(device)
            expected=gate.describe(*inputs)[0]
            gradients=torch.autograd.grad(expected.square().sum(),list(gate.readers.parameters()))
            prepared=PreparedDescriptor(gate,*inputs)
            actual=prepared()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for a,b in zip(torch.autograd.grad(actual.square().sum(),list(gate.readers.parameters())),gradients):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            with torch.no_grad():next(iter(gate.readers.parameters())).add_(.1)
            torch.testing.assert_close(prepared(),gate.describe(*inputs)[0],rtol=0,atol=0)

    def test_all_candidate_training_matches_original(self):
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            for spec in candidates():
                if not spec['reuse']:
                    stage=population()
                    a=fit_one(stage,spec,dict(FIT,steps=5),device,fast=False)
                    b=fit_one(stage,spec,dict(FIT,steps=5),device,fast=True)
                    self.assertEqual(a,b,(device,spec['key']))

if __name__=='__main__':unittest.main()
