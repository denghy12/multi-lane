import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from multi_lane.track_a.late_route_control import GateCalibration,fixed_routing,choose_training_batches,paired_bootstrap
from multi_lane.track_a.moe_activate_freeze import TaskForwardActivateFreezeBank
from multi_lane.track_a.runner import tensor_state_digest


class LateRouteTest(unittest.TestCase):
    def bank(self):
        bank=TaskForwardActivateFreezeBank(2,8,4,(11,),.1,4,top_k=2,dropout=0,noisy_gating=True,freeze_metric='gate_mass')
        bank.activate_task(0);bank.eval();bank.set_face_reliability(torch.tensor([True,False,True]))
        return bank

    def test_calibration_uses_actual_gates_and_reliable_face_only_without_state_changes(self):
        bank=self.bank();before=tensor_state_digest(bank.state_dict());capture=GateCalibration(bank)
        x=torch.randn(1,3,2,8)
        with capture.capture():
            for v in bank.view_names:bank.delta_for_layer(11,x,[0],v)
        self.assertEqual(capture.samples[0,0].tolist(),[3,3,2])
        self.assertTrue(torch.allclose(capture.mass.sum(-1)[0,0],torch.tensor([3.,3.,2.],dtype=torch.float64)))
        table,rows=capture.table();self.assertTrue(torch.equal((table>0).sum(-1),torch.full((1,1,3),2)))
        self.assertTrue(torch.allclose(table.sum(-1),torch.ones(1,1,3)))
        self.assertEqual(before,tensor_state_digest(bank.state_dict()))
        self.assertNotIn('_route',bank.__dict__)

    def test_tie_break_weights_and_empty_face_rejection(self):
        capture=GateCalibration(self.bank());capture.mass[:]=torch.tensor([2.,2.,0.,0.]);capture.samples[:]=4
        table,rows=capture.table()
        self.assertEqual(rows['0']['11']['full']['selected'],[0,1])
        self.assertEqual(table[0,0,0].tolist(),[.5,.5,0.,0.])
        capture.samples[0,0,2]=0
        with self.assertRaises(ValueError):capture.table()

    def test_fixed_routing_has_matched_weights_scale_and_restores_on_exception(self):
        bank=self.bank()
        with torch.no_grad():
            for i,expert in enumerate(bank.experts[0]):expert.up.weight.fill_(.01*(i+1))
        table=torch.zeros(1,1,3,4);table[...,0]=.25;table[...,3]=.75
        x=torch.randn(1,3,2,8);before=tensor_state_digest(bank.state_dict())
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with fixed_routing(bank,table):
                gates=bank._route(torch.randn(3,8)*100,bank.tasks[0][0],0)
                self.assertTrue(torch.equal(gates,table[0,0,0].expand(3,-1)))
                actual=bank.delta_for_layer(11,x,[0],'full')
                expected=.1*(.25*bank.experts[0][0](x[0])+.75*bank.experts[0][3](x[0]))
                self.assertTrue(torch.allclose(actual[0],expected))
                raise RuntimeError('injected')
        self.assertNotIn('_route',bank.__dict__);self.assertEqual(before,tensor_state_digest(bank.state_dict()))

    def test_calibration_grouping_excludes_future_only_images_and_keeps_people_together(self):
        class Source:
            targets=[[0],[0],[25],[5],[0,25]]
            sample_ids=['a#person=0','a#person=1','future#person=0','other#person=0','b#person=0']
        batches=choose_training_batches(Source(),0,1,20261018)
        chosen=sum(batches,[]);self.assertEqual(set(chosen),{0,1,4})
        self.assertEqual(batches,choose_training_batches(Source(),0,1,20261018))

    def test_paired_bootstrap_identical_predictions_zero_and_unpaired_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);paths={n:p/(n+'.npz') for n in ('dynamic','fixed','off')}
            data=dict(probabilities=np.array([[.9],[.2],[.8],[.1]]),targets=np.array([[1],[0],[1],[0]]),sample_ids=np.array(['a#person=0','a#person=1','b#person=0','b#person=1']),class_indices=np.array([0]))
            for path in paths.values():np.savez(path,**data)
            result=paired_bootstrap(paths,20,0)
            self.assertEqual(result['fixed']['CI95'],[0.,0.]);self.assertEqual(result['off']['point'],0.)
            data['sample_ids']=data['sample_ids'][::-1];np.savez(paths['off'],**data)
            with self.assertRaises(AssertionError):paired_bootstrap(paths,20,0)


if __name__=='__main__':unittest.main()
