import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from multi_lane.track_a.task_recovery import save_boundary,optimizer_snapshot,restore_rng,atomic_save,rng_state


class RecoveryTest(unittest.TestCase):
    def test_task_boundary_restores_model_optimizer_scheduler_rng_and_sampler(self):
        torch.manual_seed(19);random.seed(19);np.random.seed(19)
        model=nn.Sequential(nn.Linear(2,3),nn.Dropout(.2),nn.Linear(3,1))
        optimizer=torch.optim.Adam(model.parameters(),lr=.01);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.8)
        scaler=torch.cuda.amp.GradScaler(enabled=False);generator=torch.Generator().manual_seed(71)
        def step(m,o,s):
            o.zero_grad();value=m(torch.randn(4,2)).square().mean();value.backward();o.step();s.step();return value.detach().clone()
        step(model,optimizer,scheduler)
        state=optimizer_snapshot(model,optimizer,scheduler,scaler)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'task0.pth'
            save_boundary(model,model.state_dict(),0,{'git':{'commit':'fixture'}},state,generator,{'history':'done'},path)
            expected_python=random.random();expected_numpy=np.random.rand();expected_sampler=torch.randperm(8,generator=generator)
            expected_loss=step(model,optimizer,scheduler);expected=copy.deepcopy(model.state_dict())
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            new=nn.Sequential(nn.Linear(2,3),nn.Dropout(.2),nn.Linear(3,1));new.load_state_dict(loaded['model'])
            named=dict(new.named_parameters());groups=[dict(params=[named[n] for n in names]) for names in loaded['training_state']['optimizer_parameter_names']]
            opt=torch.optim.Adam(groups,lr=.01);opt.load_state_dict(loaded['training_state']['optimizer'])
            sched=torch.optim.lr_scheduler.StepLR(opt,1,.8);sched.load_state_dict(loaded['training_state']['scheduler'])
            gen=torch.Generator();gen.set_state(loaded['sampler_generator']);restore_rng(loaded['rng'])
            self.assertEqual(random.random(),expected_python);self.assertEqual(np.random.rand(),expected_numpy)
            self.assertTrue(torch.equal(torch.randperm(8,generator=gen),expected_sampler))
            self.assertTrue(torch.equal(step(new,opt,sched),expected_loss))
            self.assertTrue(all(torch.equal(v,new.state_dict()[n]) for n,v in expected.items()))
            self.assertEqual(opt.param_groups[0]['lr'],optimizer.param_groups[0]['lr'])

    def test_atomic_failure_leaves_no_final_or_partial_and_existing_snapshot_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'task0.pth'
            with patch('multi_lane.track_a.task_recovery.torch.save',side_effect=OSError('injected disk error')):
                with self.assertRaises(OSError):atomic_save({'value':1},p)
            self.assertFalse(p.exists());self.assertEqual(list(Path(tmp).iterdir()),[])
            atomic_save({'value':1},p)
            with self.assertRaises(FileExistsError):atomic_save({'value':2},p)
            self.assertEqual(torch.load(p,weights_only=False)['value'],1)

    def test_saving_does_not_advance_random_streams(self):
        model=nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        state=optimizer_snapshot(model,optimizer,scheduler,torch.cuda.amp.GradScaler(enabled=False))
        before=rng_state()
        with tempfile.TemporaryDirectory() as tmp:save_boundary(model,model.state_dict(),0,{'git':{}},state,None,{},Path(tmp)/'x.pth')
        expected=(random.random(),np.random.rand(),torch.rand(3));restore_rng(before)
        self.assertEqual(random.random(),expected[0]);self.assertEqual(np.random.rand(),expected[1]);self.assertTrue(torch.equal(torch.rand(3),expected[2]))


if __name__=='__main__':unittest.main()
