import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_track_a_reproduction as fixtures

from multi_lane.track_a.head_anchored_eight import replace_classifier_head,verify_frozen_face_predictions
from multi_lane.track_a.head_anchored_pilot import student_optimizer_groups
from multi_lane.track_a.historical_b32_backbone_attribution import method_state
from multi_lane.track_a.historical_b32_modules import modes,training_arguments,validate_arguments
from multi_lane.track_a.historical_b32_search import replace_flag
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import parse_args,tensor_state_digest,train_task
from multi_lane.track_a.selector_face_controls import (FaceSignalLoader,configure_face_only,
    face_state,install_strong_task,protected_state)


class SelectorFaceControlsTest(unittest.TestCase):
    def model(self,private=False,face=True,attention=True):
        torch.manual_seed(31)
        visual=fixtures.FakeVisual();visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            selector_mode='view_specific' if private else 'shared',adapter_mode='image_token',
            adapter_bottleneck_dim=3,adapter_layer_indices=(0,),adapter_residual_scale=.03,
            view_fusion='fixed_three_view',historical_b32_modules=True,cross_view_layer=1,
            cross_view_dim=4,cross_view_heads=2,cross_view_face_attention=face,
            **modes('ATTENTION_BOUND' if attention else 'BASE'))

    def images(self):
        return dict(full=torch.randn(3,3,4,4),person=torch.randn(3,3,4,4),face=torch.randn(3,3,4,4),
                    face_reliable=torch.tensor([True,False,True]))

    def predictions(self,model,images):
        model.eval()
        with torch.no_grad():fused,views=model.seen_logits_with_views(images)
        result=dict(sample_ids=np.asarray(['a','b','c']),targets=np.ones(tuple(fused.shape)),
                    class_indices=np.arange(fused.shape[1]),face_reliable=images['face_reliable'].numpy())
        for name,z in dict(fused=fused,**views).items():
            result[name+'_logits']=z.numpy().copy();result[name+'_probabilities']=torch.sigmoid(z).numpy()
        return result

    def test_private_copies_preserve_common_initialization_rng_and_predictions(self):
        shared=self.model();rng=torch.get_rng_state().clone();private=self.model(True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        a,b=method_state(shared),method_state(private)
        self.assertEqual(tuple(b['selectors'].shape),(2,3,2,8))
        self.assertTrue(all(torch.equal(b['selectors'][:,i],a['selectors']) for i in range(3)))
        self.assertTrue(all(torch.equal(v,b[k]) for k,v in a.items() if k!='selectors'))
        shared.activate_task(0);private.activate_task(0);images=self.images()
        self.assertTrue(torch.equal(shared.current_all_logits(images),private.current_all_logits(images)))

    def test_private_selectors_remove_direct_sharing_but_full_kv_gradient_remains(self):
        model=self.model(True);model.activate_task(0);images=self.images()
        model.cross_view_runtime_enabled=False
        _,views=model.current_all_logits_with_views(images)
        grad=torch.autograd.grad(views['face'].sum(),model.selectors)[0]
        self.assertEqual(float(grad[0,0].abs().sum()),0.)
        self.assertEqual(float(grad[0,1].abs().sum()),0.)
        self.assertGreater(float(grad[0,2].abs().sum()),0.)
        model.cross_view_runtime_enabled=True
        with torch.no_grad():model.cross_view_bank.face_tasks[0].out.weight.normal_(0,.1)
        _,views=model.current_all_logits_with_views(images)
        grad=torch.autograd.grad(views['face'].sum(),model.selectors)[0]
        self.assertGreater(float(grad[0,0].abs().sum()),0.)
        self.assertGreater(float(grad[0,2].abs().sum()),0.)
        self.assertEqual(float(grad[0,1].abs().sum()),0.)

    def test_private_banks_train_separately_and_future_task_is_unchanged(self):
        model=self.model(True);model.activate_task(0);future=model.selectors[1].detach().clone()
        with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
            train_task(model,[(self.images(),torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))],None,torch.device('cpu'),0,1,.0004,0.,1.,False,
                adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,loss_routing='adapter_asl',view_auxiliary_loss_weight=.1)
        self.assertFalse(torch.equal(model.selectors[0,0],model.selectors[0,1]))
        self.assertFalse(torch.equal(model.selectors[0,1],model.selectors[0,2]))
        self.assertTrue(torch.equal(future,model.selectors[1]))

    def test_head_swap_accepts_private_selector_shape_and_restores_on_exception(self):
        teacher=self.model(False,False,False);host=self.model(True)
        with torch.no_grad():host.head.weight.add_(.5)
        before=tensor_state_digest(method_state(host))
        with self.assertRaises(RuntimeError):
            with replace_classifier_head(host,method_state(teacher)):
                self.assertTrue(torch.equal(host.head.weight,teacher.head.weight))
                raise RuntimeError('interrupted')
        self.assertEqual(before,tensor_state_digest(method_state(host)))

    def test_frozen_base_and_prediction_invariants_after_real_face_update(self):
        person=self.model(False,False);person.activate_task(0)
        with torch.no_grad():person.cross_view_bank.tasks[0].out.weight.normal_(0,.1)
        b0=self.model(False,False,False);b0.activate_task(0)
        model=self.model();install_strong_task(model,method_state(person),method_state(b0),0)
        images=self.images();reference=self.predictions(model,images)
        protected=tensor_state_digest(protected_state(model));names=configure_face_only(model,0)
        self.assertTrue(all(n.startswith('cross_view_bank.face_tasks.0.') for n in names))
        invalid={k:v.clone() for k,v in images.items()};invalid['face_reliable'].zero_()
        y=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        loader=FaceSignalLoader([(images,y,['a','b','c']),(invalid,y,['a','b','c'])]);recovery={}
        with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
            history=train_task(model,loader,None,torch.device('cpu'),0,1,.0004,0.,1.,False,
                adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,loss_routing='joint_bce',view_auxiliary_loss_weight=.1,
                optimizer_group_builder=student_optimizer_groups,recovery_state=recovery)
        self.assertEqual(history[0]['optimizer_steps'],1)
        self.assertEqual(loader.no_signal_batches,[1])
        self.assertEqual(protected,tensor_state_digest(protected_state(model)))
        self.assertGreater(float(model.cross_view_bank.face_tasks[0].out.weight.norm()),0.)
        verify_frozen_face_predictions(self.predictions(model,images),reference,'on')
        model.cross_view_bank.face_runtime_enabled=False
        verify_frozen_face_predictions(self.predictions(model,images),reference,'face_off')
        self.assertTrue(all(n.startswith('cross_view_bank.face_tasks.0.') for group in recovery['optimizer_parameter_names'] for n in group))

    def test_new_teacher_does_not_overwrite_previously_trained_face(self):
        person=self.model(False,False);person.activate_task(0);person.activate_task(1)
        base=self.model(False,False,False);base.activate_task(0);base.activate_task(1)
        model=self.model()
        with torch.no_grad():model.cross_view_bank.face_tasks[0].out.weight.normal_(0,.2)
        before=tensor_state_digest(face_state(model))
        install_strong_task(model,method_state(person),method_state(base),1)
        self.assertEqual(before,tensor_state_digest(face_state(model)))
        configure_face_only(model,1)
        self.assertTrue(all(not p.requires_grad for p in model.cross_view_bank.face_tasks[0].parameters()))
        self.assertTrue(torch.equal(model.head.weight,base.head.weight))

    def test_private_profile_is_explicit_and_requires_face_attention(self):
        args=SimpleNamespace(data_root=Path('data'),clip_checkpoint=Path('clip'),face_manifest_root=Path('face'),smoke=True)
        flags=training_arguments(args,Path('output/control'),'ATTENTION_BOUND');replace_flag(flags,'selector_mode','view_specific')
        flags+=['--cross-view-face-attention']
        with self.assertRaises(ValueError):validate_arguments(parse_args(flags))
        validate_arguments(parse_args(flags+['--historical-b32-private-selectors']))
        with self.assertRaises(ValueError):validate_arguments(parse_args([f for f in flags if f!='--cross-view-face-attention']+['--historical-b32-private-selectors']))


if __name__=='__main__':unittest.main()
