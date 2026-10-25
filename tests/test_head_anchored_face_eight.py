import copy
import unittest
from unittest.mock import patch

import torch
import test_track_a_reproduction as fixtures

from multi_lane.track_a.cross_view_attention import TaskCrossViewAttentionBank
from multi_lane.track_a.head_anchored_eight import require_test_ready
from multi_lane.track_a.historical_b32_backbone_attribution import method_state, restore_components
from multi_lane.track_a.historical_b32_modules import modes
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import tensor_state_digest, train_task


class ReliableFaceEightTest(unittest.TestCase):
    def model(self, face=False, attention=True):
        torch.manual_seed(31)
        visual=fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            adapter_mode='image_token',adapter_bottleneck_dim=3,adapter_layer_indices=(0,),
            adapter_residual_scale=.03,view_fusion='fixed_three_view',historical_b32_modules=True,
            cross_view_layer=1,cross_view_dim=4,cross_view_heads=2,cross_view_face_attention=face,
            **modes('ATTENTION_BOUND' if attention else 'BASE'))

    def images(self):
        return dict(full=torch.randn(3,3,4,4),person=torch.randn(3,3,4,4),face=torch.randn(3,3,4,4),
                    face_reliable=torch.tensor([True,False,True]))

    def test_face_initialization_preserves_rng_common_parameters_and_zero_outputs(self):
        old=self.model();old_rng=torch.get_rng_state()
        new=self.model(True);new_rng=torch.get_rng_state()
        self.assertTrue(torch.equal(old_rng,new_rng))
        self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
        old.activate_task(0);new.activate_task(0);images=self.images()
        self.assertTrue(torch.equal(old.current_all_logits(images),new.current_all_logits(images)))
        self.assertEqual(new.cross_view_bank.parameter_count_per_task(),2*old.cross_view_bank.parameter_count_per_task())

    def test_face_residual_only_changes_reliable_rows_and_is_bounded(self):
        bank=TaskCrossViewAttentionBank(1,8,'task_full',layer_index=0,inner_dim=4,
            heads=2,scale=.1,residual_cap=.1,face_attention=True)
        with torch.no_grad():bank.face_tasks[0].out.weight.normal_(0,5.)
        face=torch.randn(1,3,2,8);full=torch.randn_like(face);selectors=torch.randn_like(face)
        reliable=torch.tensor([True,False,True])
        result=bank(face,full,selectors,[0],view_name='face',valid_mask=reliable)
        self.assertTrue(torch.equal(result[:,~reliable],face[:,~reliable]))
        self.assertGreater(float((result[:,reliable]-face[:,reliable]).norm()),0.)
        ratio=(result-face).norm(dim=-1)/face.norm(dim=-1)
        self.assertTrue(bool((ratio<=.100001).all()))
        bank.face_runtime_enabled=False
        self.assertTrue(torch.equal(face,bank(face,full,selectors,[0],view_name='face',valid_mask=reliable)))

    def test_all_unreliable_batch_skips_face_gradients_and_adam_momentum_updates(self):
        model=self.model(True);model.activate_task(0)
        valid=self.images();invalid={k:v.clone() for k,v in valid.items()}
        invalid['face_reliable'].zero_();targets=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        captures=[];original_adam=torch.optim.Adam
        class TrackedAdam(original_adam):
            def step(self,*args,**kwargs):
                result=super().step(*args,**kwargs)
                captures.append(({k:v.clone() for k,v in model.cross_view_bank.face_tasks[0].state_dict().items()},
                                 [p.grad is None for p in model.cross_view_bank.face_tasks[0].parameters()]))
                return result
        with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)),patch('multi_lane.track_a.runner.torch.optim.Adam',TrackedAdam):
            train_task(model,[(valid,targets),(invalid,targets)],None,torch.device('cpu'),0,1,.0004,0.,1.,False,
                adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,
                loss_routing='adapter_asl',view_auxiliary_loss_weight=.1)
        self.assertEqual(len(captures),2)
        self.assertTrue(all(torch.equal(v,captures[1][0][k]) for k,v in captures[0][0].items()))
        self.assertTrue(all(captures[1][1]))
        self.assertGreater(float(captures[0][0]['out.weight'].norm()),0.)

    def test_old_person_and_face_modules_freeze_together(self):
        model=self.model(True);model.activate_task(0)
        with torch.no_grad():model.cross_view_bank.face_tasks[0].out.weight.normal_(0,.1)
        protected=tensor_state_digest({k:v for k,v in model.cross_view_bank.state_dict().items() if k.split('.')[1]=='0'})
        model.activate_task(1)
        with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
            train_task(model,[(self.images(),torch.ones(3,1))],None,torch.device('cpu'),1,1,.0004,0.,1.,False,
                adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,
                loss_routing='adapter_asl',view_auxiliary_loss_weight=.1)
        self.assertEqual(protected,tensor_state_digest({k:v for k,v in model.cross_view_bank.state_dict().items() if k.split('.')[1]=='0'}))

    def test_head_restore_keeps_cotrained_representation_and_both_attention_banks(self):
        base=self.model(False,False);student=self.model(True)
        base.activate_task(0);student.activate_task(0)
        with torch.no_grad():
            student.selectors.add_(.7);student.head.weight.add_(.5)
            student.cross_view_bank.face_tasks[0].out.weight.normal_(0,.1)
        original=tensor_state_digest(method_state(student))
        asp_m=tensor_state_digest({k:v for k,v in method_state(student).items() if not k.startswith('head.')})
        with restore_components(student,method_state(base),'H'):
            self.assertTrue(torch.equal(student.head.weight,base.head.weight))
            self.assertEqual(asp_m,tensor_state_digest({k:v for k,v in method_state(student).items() if not k.startswith('head.')}))
        self.assertEqual(original,tensor_state_digest(method_state(student)))

    def test_test_requires_all_eight_completed_tasks(self):
        require_test_ready('test',False,{str(i):{} for i in range(8)})
        require_test_ready('val',True,{'0':{}})
        for smoke,files in ((True,{str(i):{} for i in range(8)}),(False,{'0':{},'2':{},'7':{}})):
            with self.assertRaises(ValueError):require_test_ready('test',smoke,files)


if __name__=='__main__':unittest.main()
