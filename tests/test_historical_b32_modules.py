import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_track_a_reproduction as fixtures
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.cross_view_attention import bounded_summary_delta
from multi_lane.track_a.spatial_summary import TaskSpatialSummary
from multi_lane.track_a.runner import (parse_args, train_task, build_optimizer_groups,
    compute_training_loss, compute_asymmetric_training_loss, compute_view_objective_losses)
from multi_lane.track_a.historical_feature_fusion import validate_arguments
from multi_lane.track_a.historical_b32_modules import ARMS, modes, training_arguments


class HistoricalB32ModulesTest(unittest.TestCase):
    def model(self, arm):
        torch.manual_seed(31)
        visual=fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            adapter_mode='image_token',adapter_bottleneck_dim=3,adapter_layer_indices=(0,),
            adapter_residual_scale=.03,view_fusion='fixed_three_view',historical_b32_modules=True,
            cross_view_layer=1,cross_view_dim=4,cross_view_heads=2,**modes(arm))

    def images(self):
        return dict(full=torch.randn(3,3,4,4),person=torch.randn(3,3,4,4),face=torch.randn(3,3,4,4),
            face_reliable=torch.tensor([True,False,True]),
            spatial_bbox=torch.tensor([[.1,.1,.8,.9],[.4,.1,.9,.9],[0.,0.,1.,1.]]),
            spatial_visibility=torch.tensor([1.,1.,0.]))

    def test_profiles_preserve_the_historical_seed1_recipe(self):
        for smoke in (True,False):
            args=SimpleNamespace(data_root=Path('data'),clip_checkpoint=Path('clip'),face_manifest_root=Path('face'),smoke=smoke)
            for arm in ARMS:
                flags=training_arguments(args,Path('output/demo'),arm)
                parsed=parse_args(flags);validate_arguments(parsed)
                self.assertEqual(parsed.seed,1);self.assertEqual(parsed.adapter_residual_scale,.03)
                self.assertEqual(parsed.view_auxiliary_loss_weight,.1)
                self.assertFalse(parsed.strong_full_asl or parsed.moe_paired_audit or parsed.module_stacking)
                self.assertEqual(parsed.amp_initial_scale,65536.)
                parsed.person_color_jitter_strength=.1
                with self.assertRaises(ValueError):validate_arguments(parsed)

    def test_zero_output_and_rng_match_across_all_arms(self):
        models={a:self.model(a) for a in ARMS};states={}
        for arm in ARMS:
            self.model(arm);states[arm]=torch.get_rng_state()
        self.assertTrue(all(torch.equal(x,states['BASE']) for x in states.values()))
        images=self.images()
        for model in models.values():model.activate_task(0)
        base=models['BASE'].current_all_logits(images)
        for arm,model in models.items():self.assertTrue(torch.equal(base,model.current_all_logits(images)),arm)

    def test_smooth_bound_is_finite_and_keeps_initial_gradient(self):
        reference=torch.randn(2,3,8,requires_grad=True)
        zero=torch.zeros_like(reference,requires_grad=True)
        bounded_summary_delta(zero,reference,.1).sum().backward()
        self.assertTrue(torch.equal(zero.grad,torch.ones_like(zero)))
        self.assertIsNone(reference.grad)
        delta=torch.randn_like(reference)*100
        value=bounded_summary_delta(delta,reference,.1)
        self.assertTrue(((value.norm(dim=-1)/reference.norm(dim=-1))<=.100001).all())
        self.assertTrue(torch.equal(bounded_summary_delta(delta,torch.zeros_like(reference),.1),torch.zeros_like(delta)))

    def test_content_calibration_is_zero_identity_and_differentiable(self):
        bank=TaskSpatialSummary(1,2,'target','content_relative')
        similarity=torch.randn(1,3,2,5,requires_grad=True);bias=torch.zeros_like(similarity,requires_grad=True)
        attention,effective=bank.attention(similarity,bias)
        self.assertTrue(torch.equal(attention,similarity.softmax(-1)))
        self.assertTrue(torch.equal(effective,bias))
        attention[...,1].sum().backward()
        self.assertGreater(float(bias.grad.norm()),0)
        self.assertGreater(float(similarity.grad.norm()),0)

    def test_actual_training_routes_auxiliary_asl_only_to_adapter(self):
        for arm in ARMS:
            model=self.model(arm);model.activate_task(0)
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if name.startswith(('adapter_bank.','cross_view_bank.','spatial_summary_bank.')):p.normal_(0,.05)
            reference=copy.deepcopy(model);images=self.images();y=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
            z,views,_=reference.current_all_logits_with_view_features(images)
            bce=compute_training_loss(z,y,[0,1],1.,'legacy_full_zero')
            asl=compute_asymmetric_training_loss(z,y,[0,1],1.,'legacy_full_zero',9.8,0.,.05,1e-8)
            vb=compute_view_objective_losses(views,images,y,[0,1],1.,'legacy_full_zero','bce')
            va=compute_view_objective_losses(views,images,y,[0,1],1.,'legacy_full_zero','asl')
            bce=bce+.1*torch.stack(list(vb.values())).mean();asl=asl+.1*torch.stack(list(va.values())).mean()
            main,adapter,_=build_optimizer_groups(reference,0,adapter_learning_rate=.0004)
            gradients=torch.autograd.grad(bce,main,retain_graph=True)+torch.autograd.grad(asl,adapter)
            expected={id(p):g for p,g in zip((*main,*adapter),gradients)}
            named={n:expected[id(p)] for n,p in reference.named_parameters() if id(p) in expected}
            with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
                train_task(model,[(images,y)],None,torch.device('cpu'),0,1,.0125,0.,1.,False,
                    adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,loss_routing='adapter_asl',
                    view_auxiliary_loss_weight=.1,loss_mode='legacy_full_zero')
            for n,p in model.named_parameters():
                if n in named:self.assertTrue(torch.equal(p.grad,named[n]),(arm,n))
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))


if __name__=='__main__':unittest.main()
