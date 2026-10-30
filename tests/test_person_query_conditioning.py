import copy
import unittest

import torch

from multi_lane.track_a.person_query_conditioning import (
    QUERY_ARMS,PersonQueryReadout,PersonQueryModel,base_arm,cap_projected_rms,
    relative_person_offset,query_configuration,trace_person_query)
from multi_lane.track_a.person_grounded_summary import GroundedReadout,MEMORIES
from multi_lane.track_a.protected_joint_attention import ProtectedJointModel,PrivateInteractionTraining
from multi_lane.track_a.protected_joint_experiment import configure_current,optimizer_groups,parser
from multi_lane.track_a.runner import TASK_SIZES,tensor_state_digest,train_task
from test_track_a_reproduction import FakeVisual


class PersonQueryTests(unittest.TestCase):
    def source(self,device='cpu'):
        torch.manual_seed(395)
        tokens={v:torch.randn(4,5,8,device=device) for v in ('full','person','face')}
        anchors={v:torch.randn(4,10,8,device=device) for v in tokens}
        return (tokens,anchors,torch.randn(4,8,device=device),torch.randn(4,8,device=device),
            torch.tensor([[0.,0.,.5,1.],[.5,0.,1.,1.],[0.,0.,1.,1.],[2.,2.,3.,3.]],device=device),
            torch.tensor([1.,1.,.4,0.],device=device),torch.tensor([True,False,True,False],device=device))

    def model(self,arm):
        torch.manual_seed(293)
        cls=ProtectedJointModel if arm=='C0' else PersonQueryModel
        model=cls(FakeVisual(),TASK_SIZES,num_selectors=10,num_prompts=2,num_prompt_layers=1,
            view_fusion='fixed_three_view',adapter_mode='image_token',adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,),adapter_residual_scale=.03)
        before=torch.get_rng_state();model.initialize_interaction(arm,layer=0,dim=4,heads=2)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        model.activate_task(0);configure_current(model,0)
        x={v:torch.randn(4,3,4,4) for v in ('full','person','face')}
        x.update(face_reliable=torch.tensor([True,False,True,False]),spatial_visibility=torch.tensor([1.,1.,.4,0.]),
            spatial_bbox=torch.tensor([[0.,0.,.5,1.],[.5,0.,1.,1.],[0.,0.,1.,1.],[2.,2.,3.,3.]]))
        y=torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],[0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]])
        return model,(x,y)

    def train(self,model,batch):
        return train_task(model,[batch],None,torch.device('cpu'),0,3,.001,0.,1.,False,
            adapter_learning_rate=.0004,loss_routing='adapter_asl',view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups,
            private_interaction=PrivateInteractionTraining(model,epochs=3,amp=False) if model.interaction_bank else None)

    def test_projection_ceiling_identity_direction_and_zero_gradient(self):
        x=torch.tensor([[.1,-.2,.3,-.4],[100.,-200.,300.,-400.],[0.,0.,0.,0.]],requires_grad=True)
        y=cap_projected_rms(x)
        self.assertTrue(torch.equal(x[0],y[0]))
        self.assertLessEqual(float(y.square().mean(-1).sqrt().max()),1.000001)
        self.assertTrue(torch.allclose(torch.nn.functional.cosine_similarity(x[1:],y[1:]),torch.tensor([1.,0.])))
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[-1].norm()),0)

    def test_relative_gain_has_safe_zero_initialization_and_fixed_upper_bound(self):
        raw=torch.zeros(2,2,8,requires_grad=True);anchor=torch.ones_like(raw)
        y=relative_person_offset(raw,anchor);self.assertEqual(float(y.abs().sum()),0)
        y.sum().backward();self.assertTrue(torch.allclose(raw.grad,torch.full_like(raw,32.)))
        small=torch.full_like(raw,1e-9);self.assertTrue(torch.allclose(relative_person_offset(small,anchor),32*small))
        large=torch.full_like(raw,100.)
        self.assertLessEqual(float(relative_person_offset(large,anchor).square().mean(-1).sqrt().max()),.010001)
        self.assertEqual(float(relative_person_offset(large,torch.zeros_like(anchor)).abs().sum()),0)

    def test_policies_have_original_initialization_and_no_added_parameters(self):
        for arm in QUERY_ARMS:
            torch.manual_seed(930100);old=GroundedReadout(768,512,10,base_arm(arm))
            torch.manual_seed(930100);new=PersonQueryReadout(768,512,10,arm)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
            self.assertEqual(sum(p.numel() for p in new.parameters()),221728 if arm.endswith('ATTN') else 221789)
            cfg=query_configuration(arm);self.assertFalse(cfg['combined_repairs'])
            self.assertEqual(cfg['new_trainable_parameters'],0)
            self.assertEqual(cfg['localization_weight'],.01 if arm.startswith('QPROJ') else 0.)

    def test_projection_and_gain_are_mutually_exclusive_single_factors(self):
        x=self.source()
        for arm in ('QPROJ_GROUND_ATTN','QGAIN_CONTENT_ATTN'):
            m=PersonQueryReadout(8,8,10,arm,4,2)
            with torch.no_grad():m.condition_input.weight.mul_(1000);m.condition_output.normal_()
            c=m.condition_components(x[1]['full'],x[3])
            if arm.startswith('QPROJ'):
                self.assertLessEqual(float(c['preactivation'].square().mean(-1).sqrt().max()),1.000001)
                self.assertTrue(torch.equal(c['raw_shift'],c['shift']))
                self.assertGreater(float((1-c['condition'].square()).mean()),.1)
            else:
                self.assertTrue(torch.equal(c['raw_preactivation'],c['preactivation']))
                ratio=c['shift'].square().mean(-1).sqrt()/c['anchor'].square().mean(-1).sqrt()
                self.assertLessEqual(float(ratio.max()),.010001)

    def test_inactive_projection_bound_exactly_replays_original_readout(self):
        x=self.source();m=PersonQueryReadout(8,8,10,'QPROJ_GROUND_ATTN',4,2)
        with torch.no_grad():m.condition_input.weight.mul_(.001);m.condition_output.normal_();m.out.weight.normal_()
        old=GroundedReadout(8,8,10,'PERSON_GROUND_ATTN',4,2);old.load_state_dict(m.state_dict())
        a=m.summarize(x[0],x[1],x[3],x[4],x[5]);b=old.summarize(x[0],x[1],x[3],x[4],x[5])
        for k in MEMORIES:self.assertTrue(torch.equal(a[k],b[k]),k)
        self.assertTrue(torch.equal(m(x[2],x[3],a,x[6]),old(x[2],x[3],b,x[6])))

    def test_own_wrong_and_zero_condition_traces_keep_other_evidence_fixed(self):
        x=self.source();donor=torch.tensor([1,0,3,2])
        for arm in QUERY_ARMS:
            m=PersonQueryReadout(8,8,10,arm,4,2)
            with torch.no_grad():m.condition_output.normal_();m.out.weight.normal_()
            a,sa=trace_person_query(m,*x);b,sb=trace_person_query(m,*x,condition_person=x[3][donor])
            self.assertFalse(torch.equal(a['query'],b['query']))
            self.assertTrue(torch.equal(a['readout_query'],b['readout_query']))
            for k in ('global','person','face'):self.assertTrue(torch.equal(sa[k],sb[k]))
            az,_=trace_person_query(m,*x,zero_condition=True)
            bz,_=trace_person_query(m,*x,condition_person=x[3][donor],zero_condition=True)
            self.assertTrue(torch.equal(az['query'],bz['query']));self.assertTrue(torch.equal(az['delta'],bz['delta']))

    def test_original_training_rng_and_predictions_remain_exact(self):
        base,batch=self.model('C0');reference=self.train(base,batch)
        state=lambda m:{k:v for k,v in m.state_dict().items() if not k.startswith('interaction_bank.')}
        expected=tensor_state_digest(state(base));rng=torch.get_rng_state()
        for arm in QUERY_ARMS:
            m,batch=self.model(arm);m.eval()
            with torch.no_grad():
                on,_=m.seen_logits_with_views(batch[0]);self.assertTrue(torch.equal(on,m.cached_base_seen_logits()))
            rows=self.train(m,batch)
            self.assertEqual(expected,tensor_state_digest(state(m)),arm)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual([x['current_loss'] for x in rows],[x['current_loss'] for x in reference])
            self.assertTrue(all(x['interaction_skipped_steps']==0 for x in rows))

    def test_private_objective_leaves_original_and_head_gradients_empty(self):
        for arm in QUERY_ARMS:
            m,(x,y)=self.model(arm);m.train();m.current_all_logits_with_views(x)
            private=PrivateInteractionTraining(m,epochs=3,amp=False);private.begin_epoch();private.backward(y,range(5))
            self.assertTrue(any(p.grad is not None and float(p.grad.norm())>0 for p in m.interaction_parameters()))
            self.assertTrue(all(p.grad is None for k,p in m.named_parameters() if not k.startswith('interaction_bank.')))

    def test_roundtrip_and_final_feature_cap(self):
        x=self.source()
        for arm in QUERY_ARMS:
            m=PersonQueryReadout(8,8,10,arm,4,2)
            with torch.no_grad():m.condition_output.normal_();m.out.weight.normal_()
            a,_=trace_person_query(m,*x)
            clone=PersonQueryReadout(8,8,10,arm,4,2);clone.load_state_dict(copy.deepcopy(m.state_dict()))
            b,_=trace_person_query(clone,*x)
            self.assertTrue(torch.equal(a['delta'],b['delta']))
            ratio=a['delta'].square().mean(-1).sqrt()/x[2].square().mean(-1).sqrt()
            self.assertLessEqual(float(ratio.max()),.010001)

    def test_cli_accepts_all_new_arm_names(self):
        for arm in QUERY_ARMS:
            args=parser().parse_args(['train','--arm',arm,'--output','output/example'])
            self.assertEqual(args.arm,arm)

    @unittest.skipUnless(torch.cuda.is_available(),'actual CUDA AMP required')
    def test_actual_cuda_amp_keeps_new_query_path_fp32(self):
        x=self.source('cuda')
        for arm in QUERY_ARMS:
            m=PersonQueryReadout(8,8,10,arm,4,2).cuda()
            with torch.cuda.amp.autocast(enabled=True):trace,_=trace_person_query(m,*x)
            for k in ('preactivation','condition','shift','query','alpha','delta'):
                self.assertEqual(trace[k].dtype,torch.float32,k)


if __name__=='__main__':unittest.main()
