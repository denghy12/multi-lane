import copy
import unittest

import torch

from multi_lane.track_a.protected_attention_dose import (
    DOSE_ARMS, DoseMatchedModel, DoseMatchedReadout, dose_configuration)
from multi_lane.track_a.protected_joint_attention import ProtectedJointModel, PrivateInteractionTraining
from multi_lane.track_a.protected_joint_experiment import configure_current, optimizer_groups
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest, train_task
from multi_lane.track_a.task_recovery import rng_state, restore_rng
from test_track_a_reproduction import FakeVisual


class DoseMatchedTests(unittest.TestCase):
    def make(self,arm):
        torch.manual_seed(281)
        cls=ProtectedJointModel if arm=='C0' else DoseMatchedModel
        m=cls(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,num_prompt_layers=1,
            view_fusion='fixed_three_view',adapter_mode='image_token',adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,),adapter_residual_scale=.03)
        before=torch.get_rng_state();m.initialize_interaction(arm,layer=0,dim=4,heads=2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        m.activate_task(0);configure_current(m,0)
        x={v:torch.randn(4,3,4,4) for v in ('full','person','face')}
        x.update(face_reliable=torch.tensor([True,False,True,False]))
        y=torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],
                        [0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]])
        return m,(x,y)

    def train(self,m,batch,**kwargs):
        return train_task(m,[batch],None,torch.device('cpu'),0,3,.001,0.,1.,False,
            adapter_learning_rate=.0004,loss_routing='adapter_asl',view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups,
            private_interaction=PrivateInteractionTraining(m,epochs=3,amp=False) if m.interaction_bank else None,
            **kwargs)

    def base(self,m):
        return {k:v for k,v in m.state_dict().items() if not k.startswith('interaction_bank.')}

    def test_initial_gain_one_identity_and_attention_qkv_paired_for_all_tasks(self):
        expected=None
        for arm in DOSE_ARMS:
            m,(x,y)=self.make(arm);m.eval()
            with torch.no_grad():
                z,_=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(z,m.cached_base_seen_logits()))
            self.assertEqual(m.interaction_bank[0].last_metrics['gate_mean'],1.)
            if arm!='DOSE_MLP':
                state=[{k:v for k,v in b.state_dict().items()
                        if k.startswith(('query.','key.','value.','out.','view_types'))} for b in m.interaction_bank]
                if expected is None:expected=state
                else:
                    for a,b in zip(expected,state):
                        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            self.assertNotIn(m.interaction_arm,('C3','C4','FULL_ONLY'))

    def test_all_gains_and_raw_scales_share_exact_final_one_percent_upper_bound(self):
        base=torch.randn(5,8);base[0]=0
        raw=torch.randn(5,8)*100
        for arm in DOSE_ARMS:
            module=DoseMatchedReadout(8,8,arm,dim=4,heads=2)
            for gain in (0.,.01,1.,2.,1e5):
                out=module._correction(raw,base,torch.full((5,1),gain))
                ratio=out.square().mean(-1).sqrt()/base.square().mean(-1).sqrt().clamp_min(1e-12)
                self.assertTrue(bool((ratio<=.010001).all()))
                self.assertTrue(bool(torch.isfinite(out).all()))
                self.assertEqual(int(out[0].count_nonzero()),0)

    def test_real_updates_preserve_original_base_losses_and_rng(self):
        m,batch=self.make('C0');reference=self.train(m,batch)
        state=tensor_state_digest(self.base(m));rng=torch.get_rng_state()
        for arm in DOSE_ARMS:
            m,batch=self.make(arm);rows=self.train(m,batch)
            self.assertEqual(state,tensor_state_digest(self.base(m)),arm)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),arm)
            self.assertEqual([r['current_loss'] for r in reference],[r['current_loss'] for r in rows])
            self.assertGreater(rows[-1]['interaction_gradient_norm'],0)
            self.assertGreater(rows[-1]['interaction_delta_to_base_rms'],0)
            self.assertLessEqual(rows[-1]['interaction_delta_to_base_rms_max'],.010001)
            if arm=='DOSE_TASK':self.assertGreater(float(m.interaction_bank[0].task_gate_logit.abs()),0.)

    def test_task_scalar_constant_across_people_and_content_gain_can_depend_on_input(self):
        b=torch.randn(4,8);p=torch.randn_like(b);face=torch.randn_like(b)
        valid=torch.tensor([True,False,True,False]);visible=torch.ones(4,dtype=torch.bool)
        task=DoseMatchedReadout(8,8,'DOSE_TASK',4,2)
        task.task_gate_logit.data.fill_(.4)
        a=task._gate_value(b,b,p,face,valid,visible)
        self.assertEqual(float(a.std()),0.)
        content=DoseMatchedReadout(8,8,'DOSE_CONTENT',4,2)
        with torch.no_grad():
            content.gate[-1].weight.fill_(.3)
        a=content._gate_value(b,b,p,face,valid,visible)
        self.assertGreater(float(a.std()),0.)

    def test_private_gradient_has_no_base_head_or_old_task_recipient(self):
        for arm in DOSE_ARMS:
            m,(x,y)=self.make(arm);m.train();m.current_all_logits_with_views(x)
            private=PrivateInteractionTraining(m,amp=False);private.begin_epoch();private.backward(y,range(5))
            self.assertTrue(all(p.grad is None for n,p in m.named_parameters()
                                if not n.startswith('interaction_bank.0.')))
            main,adapter,_=optimizer_groups(model=m,weight_decay=0.,adapter_learning_rate=.0004,
                adapter_weight_decay=0.,selector_condition_learning_rate=.0004,view_fusion_learning_rate=.0004)
            self.assertFalse({id(p) for p in private.parameters}&{id(p) for p in main+adapter})

    def test_invalid_face_and_off_predictions_preserved_after_training(self):
        for arm in DOSE_ARMS:
            m,(x,y)=self.make(arm);self.train(m,(x,y));m.eval()
            with torch.no_grad():
                z,views=m.seen_logits_with_views(x);off=m.cached_base_seen_logits()
                changed=copy.deepcopy(x);changed['face'][~x['face_reliable']]*=17
                other,_=m.seen_logits_with_views(changed)
                self.assertTrue(torch.equal(z[~x['face_reliable']],other[~x['face_reliable']]))
                m.interaction_enabled=False
                zoff,v=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(off,zoff))
                self.assertTrue(all(torch.equal(views[k],v[k]) for k in v))

    def test_epoch_recovery_restores_gates_optimizers_and_exact_trajectory(self):
        for arm in DOSE_ARMS:
            m,batch=self.make(arm);saved={}
            def capture(epoch,rows,state):
                saved[epoch]=(copy.deepcopy(m.state_dict()),copy.deepcopy(state),rng_state())
            rows=self.train(m,batch,epoch_callback=capture);expected=tensor_state_digest(m.state_dict())
            n,batch=self.make(arm);weights,state,rng=saved[1];n.load_state_dict(weights);restore_rng(rng)
            actual=self.train(n,batch,resume_training_state=state)
            self.assertEqual(expected,tensor_state_digest(n.state_dict()))
            for key in ('current_loss','interaction_loss','interaction_gate_mean'):
                self.assertEqual([r[key] for r in rows],[r[key] for r in actual])

    def test_manifest_distinguishes_final_bound_from_old_post_bound_gate(self):
        for arm in DOSE_ARMS:
            cfg=dose_configuration(arm)
            self.assertEqual(cfg['final_residual_cap'],.01)
            self.assertFalse(cfg['target_region_constraint'])
            self.assertFalse(cfg['new_private_summarize_queries'])
            self.assertEqual(cfg['initial_gain'],1.)
            self.assertEqual(cfg['cap_placement'],'after_gain_times_raw_before_h0_addition')


if __name__=='__main__':unittest.main()
