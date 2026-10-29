import copy
import unittest
import torch

from multi_lane.track_a.private_summarize_attention import (
    SUMMARY_ARMS,PrivateSummaryModel,PrivateSummaryReadout,norm_matched_query,summary_configuration)
from multi_lane.track_a.private_summary_evaluation import (grouped_donors,paired_batches,match_direction_magnitude)
from multi_lane.track_a.protected_joint_attention import ProtectedJointModel,PrivateInteractionTraining
from multi_lane.track_a.protected_joint_experiment import configure_current,optimizer_groups
from multi_lane.track_a.runner import TASK_SIZES,tensor_state_digest,train_task
from multi_lane.track_a.task_recovery import rng_state,restore_rng
from test_track_a_reproduction import FakeVisual


class PrivateSummaryTests(unittest.TestCase):
    def make(self,arm):
        torch.manual_seed(281)
        cls=ProtectedJointModel if arm=='C0' else PrivateSummaryModel
        m=cls(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,num_prompt_layers=1,
            view_fusion='fixed_three_view',adapter_mode='image_token',adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,),adapter_residual_scale=.03)
        before=torch.get_rng_state();m.initialize_interaction(arm,layer=0,dim=4,heads=2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        m.activate_task(0);configure_current(m,0)
        x={v:torch.randn(4,3,4,4) for v in ('full','person','face')}
        x.update(face_reliable=torch.tensor([True,False,True,False]),spatial_visibility=torch.ones(4))
        y=torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],[0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]])
        return m,(x,y)

    def train(self,m,batch,**kwargs):
        return train_task(m,[batch],None,torch.device('cpu'),0,3,.001,0.,1.,False,
            adapter_learning_rate=.0004,loss_routing='adapter_asl',view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups,
            private_interaction=PrivateInteractionTraining(m,epochs=3,amp=False) if m.interaction_bank else None,
            **kwargs)

    def base(self,m):return {k:v for k,v in m.state_dict().items() if not k.startswith('interaction_bank.')}

    def test_norm_match_and_both_upstream_gradient_boundaries(self):
        anchor=torch.randn(4,2,8,requires_grad=True);offset=torch.randn(2,8,requires_grad=True)
        q=norm_matched_query(anchor,offset)
        self.assertTrue(torch.allclose(q.norm(dim=-1),anchor.norm(dim=-1),atol=1e-6))
        (q*torch.randn_like(q)).sum().backward()
        self.assertIsNone(anchor.grad);self.assertGreater(float(offset.grad.norm()),0)
        self.assertTrue(torch.equal(norm_matched_query(anchor,torch.zeros_like(offset)),anchor.detach()))
        self.assertTrue(torch.equal(norm_matched_query(torch.zeros_like(anchor),offset),torch.zeros_like(anchor)))
        self.assertTrue(torch.isfinite(norm_matched_query(anchor,-anchor.detach())).all())

    def test_common_parameters_and_zero_queries_replay_across_all_tasks(self):
        expected=None
        for arm in SUMMARY_ARMS:
            m,(x,y)=self.make(arm);m.eval()
            state=[{k:v for k,v in module.state_dict().items() if k.startswith(('query.','value.','out.','view_types'))} for module in m.interaction_bank]
            if expected is None:expected=state
            else:
                for a,b in zip(expected,state):self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            with torch.no_grad():
                z,_=m.seen_logits_with_views(x);self.assertTrue(torch.equal(z,m.cached_base_seen_logits()))
                self.assertTrue(torch.equal(m.interaction_delta(),m.interaction_delta(zero_query=True)))
            self.assertEqual(m.interaction_bank[0].query_offset.requires_grad,'LEARN' in arm)

    def test_real_updates_keep_original_parameters_losses_rng_and_learn_queries(self):
        m,batch=self.make('C0');reference=self.train(m,batch);expected=tensor_state_digest(self.base(m));rng=torch.get_rng_state()
        for arm in SUMMARY_ARMS:
            m,batch=self.make(arm);rows=self.train(m,batch)
            self.assertEqual(expected,tensor_state_digest(self.base(m)),arm)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),arm)
            self.assertEqual([v['current_loss'] for v in rows],[v['current_loss'] for v in reference])
            self.assertGreater(rows[-1]['interaction_gradient_norm'],0)
            self.assertLessEqual(rows[-1]['interaction_delta_to_base_rms_max'],.010001)
            if 'LEARN' in arm:
                self.assertGreater(rows[-1]['interaction_query_gradient_norm'],0)
                self.assertGreater(float(m.interaction_bank[0].query_offset.detach().norm()),0)
            else:self.assertEqual(rows[-1]['interaction_query_gradient_norm'],0)

    def test_summarization_gradients_reach_only_new_queries(self):
        module=PrivateSummaryReadout(8,8,2,'SUMMARY_LEARN_ATTN',4,2)
        tokens={v:torch.randn(4,5,8,requires_grad=True) for v in ('full','person','face')}
        anchors={v:torch.randn(4,2,8,requires_grad=True) for v in tokens}
        summaries=module.summarize(tokens,anchors)
        sum((s*torch.randn_like(s)).sum() for s in summaries.values()).backward()
        self.assertGreater(float(module.query_offset.grad.norm()),0)
        self.assertTrue(all(t.grad is None for t in (*tokens.values(),*anchors.values())))

    def test_private_backward_has_no_head_or_original_parameter_gradients(self):
        for arm in SUMMARY_ARMS:
            m,(x,y)=self.make(arm);m.train();m.current_all_logits_with_views(x)
            private=PrivateInteractionTraining(m,amp=False);private.begin_epoch();private.backward(y,range(5))
            self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('interaction_bank.0.')))

    def test_invalid_face_memory_and_off_invariance_after_updates(self):
        for arm in SUMMARY_ARMS:
            m,(x,y)=self.make(arm);self.train(m,(x,y));m.eval()
            with torch.no_grad():
                z,views=m.seen_logits_with_views(x);off=m.cached_base_seen_logits();native=m.interaction_delta()
                no_face=m.interaction_delta(no_face=True)
                self.assertTrue(torch.equal(native[~x['face_reliable']],no_face[~x['face_reliable']]))
                changed=copy.deepcopy(x);changed['face'][~x['face_reliable']]*=13
                a,_=m.seen_logits_with_views(changed)
                self.assertTrue(torch.equal(z[~x['face_reliable']],a[~x['face_reliable']]))
                m.interaction_enabled=False;b,v=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(off,b));self.assertTrue(all(torch.equal(views[k],v[k]) for k in v))

    def test_epoch_recovery_includes_queries_and_both_optimizers(self):
        for arm in SUMMARY_ARMS:
            m,batch=self.make(arm);saved={}
            def capture(epoch,rows,state):saved[epoch]=(copy.deepcopy(m.state_dict()),copy.deepcopy(state),rng_state())
            rows=self.train(m,batch,epoch_callback=capture);expected=tensor_state_digest(m.state_dict())
            n,batch=self.make(arm);weights,state,rng=saved[1];n.load_state_dict(weights);restore_rng(rng)
            actual=self.train(n,batch,resume_training_state=state)
            self.assertEqual(expected,tensor_state_digest(n.state_dict()))
            for key in ('current_loss','interaction_loss','interaction_query_gradient_norm'):
                self.assertEqual([v[key] for v in rows],[v[key] for v in actual])

    def test_full_size_parameter_matching_and_query_count(self):
        expected=(198768,198912,206448,206592)
        for arm,count in zip(SUMMARY_ARMS,expected):
            m=PrivateSummaryReadout(768,512,10,arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()),count)
            self.assertEqual(summary_configuration(arm)['query_parameters_per_task'],7680 if 'LEARN' in arm else 0)
        self.assertEqual(8*10*768,61440)

    def test_same_image_donors_visibility_and_batch_integrity(self):
        ids=['a#person=0','b#person=0','a#person=1','c#person=0','a#person=2','b#person=1']
        donor,mask=grouped_donors(ids,torch.tensor([1,1,1,1,0,0],dtype=torch.bool))
        self.assertEqual(mask.tolist(),[True,False,True,False,False,False])
        self.assertEqual(donor.tolist(),[2,1,0,3,4,5])
        self.assertEqual(paired_batches(ids),[[0,2,4,1,5]])

    def test_direction_interventions_keep_recipient_magnitude(self):
        native=torch.randn(5,3,8);direction=torch.randn(1,3,8)
        z=match_direction_magnitude(native,direction)
        self.assertTrue(torch.allclose(z.norm(dim=-1),native.norm(dim=-1),atol=1e-6))
        self.assertTrue(torch.equal(match_direction_magnitude(native,torch.zeros_like(direction)),native))

    def test_memory_intervention_holds_query_inputs_fixed(self):
        m,(x,y)=self.make('SUMMARY_LEARN_ATTN');self.train(m,(x,y));m.eval()
        with torch.no_grad():
            m.seen_logits_with_views(x);q=[]
            hook=m.interaction_bank[0].query.register_forward_pre_hook(lambda _,args:q.append(args[0].clone()))
            m.interaction_delta();m.interaction_delta(no_face=True);m.interaction_delta(no_person=True)
            hook.remove()
            self.assertTrue(torch.equal(q[0],q[1]) and torch.equal(q[0],q[2]))


if __name__=='__main__':unittest.main()
