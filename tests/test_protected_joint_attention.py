import copy
import unittest

import torch

from multi_lane.track_a.protected_joint_attention import (
    ARMS, ProtectedJointModel, PrivateInteractionTraining, target_summary)
from multi_lane.track_a.protected_joint_experiment import configure_current, optimizer_groups
from multi_lane.track_a.runner import TASK_SIZES, train_task, tensor_state_digest
from multi_lane.track_a.task_recovery import rng_state, restore_rng
from test_track_a_reproduction import FakeVisual


class ProtectedInteractionTests(unittest.TestCase):
    def make(self, arm='C4'):
        torch.manual_seed(281)
        m = ProtectedJointModel(FakeVisual(), TASK_SIZES, num_selectors=2,
            num_prompts=2,num_prompt_layers=1,view_fusion='fixed_three_view',
            adapter_mode='image_token',adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,),adapter_residual_scale=.03)
        before = torch.get_rng_state()
        m.initialize_interaction(arm,layer=0,dim=4,heads=2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        m.activate_task(0); configure_current(m,0)
        x = {v:torch.randn(4,3,4,4) for v in ('full','person','face')}
        x.update(face_reliable=torch.tensor([True,False,True,False]),
            spatial_bbox=torch.tensor([[0.,0.,.5,1.]]*4),spatial_visibility=torch.ones(4))
        y = torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],
                          [0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]])
        return m,(x,y)

    def train(self,m,batch,epochs=3,**kwargs):
        return train_task(m,[batch],None,torch.device('cpu'),0,epochs,.001,0.,1.,False,
            adapter_learning_rate=.0004,loss_routing='adapter_asl',view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups,
            private_interaction=PrivateInteractionTraining(m,epochs=epochs,amp=False)
                if m.interaction_bank else None,**kwargs)

    def base_state(self,m):
        return {k:v for k,v in m.state_dict().items() if not k.startswith('interaction_bank.')}

    def test_live_branch_exactly_preserves_base_updates_rng_and_losses(self):
        m,batch=self.make('C0'); base=self.train(m,batch)
        expected=tensor_state_digest(self.base_state(m)); rng=torch.get_rng_state()
        for arm in ARMS[1:]:
            m,batch=self.make(arm); rows=self.train(m,batch)
            self.assertEqual(tensor_state_digest(self.base_state(m)),expected,arm)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),arm)
            self.assertEqual([r['current_loss'] for r in base],[r['current_loss'] for r in rows])
            self.assertGreater(rows[-1]['interaction_gradient_norm'],0)
            self.assertGreater(rows[-1]['interaction_delta_to_base_rms'],0)

    def test_zero_init_off_and_invalid_face_prediction_invariance(self):
        for arm in ARMS[1:]:
            m,(x,y)=self.make(arm); m.eval()
            with torch.no_grad():
                z,_=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(z,m.cached_base_seen_logits()),arm)
            self.train(m,(x,y)); m.eval()
            with torch.no_grad():
                z,vs=m.seen_logits_with_views(x)
                off=m.cached_base_seen_logits()
                self.assertGreater(float((z-off).abs().max()),0)
                changed=copy.deepcopy(x); changed['face'][~x['face_reliable']]*=9
                a,_=m.seen_logits_with_views(changed)
                self.assertTrue(torch.equal(z[~x['face_reliable']],a[~x['face_reliable']]),arm)
                m.interaction_enabled=False
                zoff,vs_off=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(off,zoff),arm)
                for v in vs:
                    self.assertTrue(torch.equal(vs[v],vs_off[v]),arm)

    def test_target_mask_ignores_cls_outside_patches_and_invisible_target(self):
        tokens=torch.randn(2,5,8); selectors=torch.randn(1,2,2,8)
        box=torch.tensor([[0.,0.,.49,.49],[2.,2.,3.,3.]])
        a,valid=target_summary(tokens,selectors,box,torch.ones(2))
        self.assertEqual(valid.tolist(),[True,False])
        self.assertTrue(torch.equal(a[0,0],tokens[0,1].expand(2,-1)))
        self.assertEqual(int(a[0,1].count_nonzero()),0)
        tokens[:,0]*=100; tokens[:,2:]*=5
        b,_=target_summary(tokens,selectors,box,torch.ones(2))
        self.assertTrue(torch.equal(a,b))

    def test_private_gradient_cannot_touch_base_and_optimizer_membership_disjoint(self):
        m,(x,y)=self.make(); m.train(); m.current_all_logits_with_views(x)
        p=PrivateInteractionTraining(m,amp=False); p.begin_epoch(); p.backward(y,list(range(5)))
        private={id(v) for v in p.parameters}
        self.assertTrue(all(v.grad is None for n,v in m.named_parameters()
                            if not n.startswith('interaction_bank.')))
        main,adapter,_=optimizer_groups(model=m,weight_decay=0.,adapter_learning_rate=.0004,
            adapter_weight_decay=0.,selector_condition_learning_rate=.0004,view_fusion_learning_rate=.0004)
        self.assertFalse(private & {id(v) for v in main+adapter})

    def test_epoch_resume_restores_both_optimizers_and_branch(self):
        m,batch=self.make(); saved={}
        def capture(epoch,rows,state):
            saved[epoch]=(copy.deepcopy(m.state_dict()),copy.deepcopy(state),rng_state())
        rows=self.train(m,batch,epoch_callback=capture)
        expected=tensor_state_digest(m.state_dict())
        n,batch=self.make(); weights,state,rng=saved[1]
        n.load_state_dict(weights); restore_rng(rng)
        recovered=self.train(n,batch,resume_training_state=state)
        self.assertEqual(expected,tensor_state_digest(n.state_dict()))
        for key in ('current_loss','interaction_loss','interaction_gradient_norm'):
            self.assertEqual([r[key] for r in rows],[r[key] for r in recovered])


if __name__=='__main__':
    unittest.main()
