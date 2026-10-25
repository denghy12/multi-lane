import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_track_a_reproduction as fixtures

from multi_lane.track_a.head_anchored_pilot import (
    ARMS, adaptation_rng, configure_student, install_current_teacher,
    load_initial_teacher, rng_digest, student_optimizer_groups, task_slices)
from multi_lane.track_a.historical_b32_backbone_attribution import method_state
from multi_lane.track_a.historical_b32_modules import modes
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import (
    compute_asymmetric_training_loss, compute_training_loss, compute_view_objective_losses,
    tensor_state_digest, train_task)
from multi_lane.track_a.task_recovery import rng_state


class HeadAnchoredPilotTest(unittest.TestCase):
    def model(self, attention=True):
        torch.manual_seed(31)
        visual = fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual, (2,1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode='image_token', adapter_bottleneck_dim=3,
            adapter_layer_indices=(0,), adapter_residual_scale=.03, view_fusion='fixed_three_view',
            historical_b32_modules=True, cross_view_layer=1, cross_view_dim=4, cross_view_heads=2,
            **modes('ATTENTION_BOUND' if attention else 'BASE'))

    def images(self):
        return dict(full=torch.randn(3,3,4,4), person=torch.randn(3,3,4,4), face=torch.randn(3,3,4,4),
                    face_reliable=torch.tensor([True,False,True]))

    def test_teacher_loading_preserves_old_students_and_uses_teacher_current_coordinates(self):
        teacher = self.model(False); teacher.activate_task(0)
        with torch.no_grad():
            teacher.selectors[0].add_(.3); teacher.head.weight[:2].add_(.5)
        initial = {k:v.clone() for k,v in method_state(teacher).items()}
        student = self.model(); load_initial_teacher(student, initial)
        with torch.no_grad(): student.selectors[0].add_(1.)
        old = student.selectors[0].clone()
        teacher.activate_task(1)
        source = {k:v.clone() for k,v in method_state(teacher).items()}
        before = tensor_state_digest(source)
        audit = install_current_teacher(student, source, 1)
        self.assertTrue(audit['old_student_coordinates_preserved'])
        self.assertTrue(torch.equal(student.selectors[0], old))
        self.assertTrue(torch.equal(student.selectors[1], teacher.selectors[1]))
        self.assertFalse(torch.equal(student.selectors[1], student.selectors[0]))
        self.assertEqual(before, tensor_state_digest(source))

    def test_all_arms_route_gradients_and_keep_frozen_coordinates_exact(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = self.model(arm != 'HEAD_FIXED_NO_ATTENTION'); model.activate_task(0)
                # Nonzero initial A/M permits gradients to their down/q/k/v layers.
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name.startswith(('adapter_bank.', 'cross_view_bank.')): p.normal_(0,.05)
                configure_student(model, arm, 0)
                initial_head = {k:v.clone() for k,v in model.head.state_dict().items()}
                initial = {k:v.clone() for k,v in method_state(model).items()}
                reference = copy.deepcopy(model)
                images = self.images(); targets = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
                logits, views = reference.current_all_logits_with_views(images)
                bce = compute_training_loss(logits, targets, [0,1], 1., 'legacy_full_zero')
                bce += .1*torch.stack(list(compute_view_objective_losses(
                    views, images, targets, [0,1], 1., 'legacy_full_zero', 'bce').values())).mean()
                asl = compute_asymmetric_training_loss(logits, targets, [0,1], 1., 'legacy_full_zero')
                asl += .1*torch.stack(list(compute_view_objective_losses(
                    views, images, targets, [0,1], 1., 'legacy_full_zero', 'asl').values())).mean()
                main, adapter, groups = student_optimizer_groups(model=reference, weight_decay=0.,
                    adapter_learning_rate=.0004, view_fusion_learning_rate=.0004)
                gradients = torch.autograd.grad(bce, main, retain_graph=bool(adapter))
                if adapter: gradients += torch.autograd.grad(asl, adapter)
                expected = {id(p):g for p,g in zip(main+adapter, gradients)}
                expected_named = {n:expected[id(p)] for n,p in reference.named_parameters() if id(p) in expected}
                recovery = {}
                with patch('multi_lane.track_a.runner.TASK_SIZES', (2,1)):
                    train_task(model, [(images, targets)], None, torch.device('cpu'), 0, 1, .0004, 0., 1., False,
                        adapter_learning_rate=.0004, view_fusion_learning_rate=.0004,
                        view_auxiliary_loss_weight=.1,
                        loss_routing='joint_bce' if arm == 'BASE_FIXED_ATTENTION' else 'adapter_asl',
                        optimizer_group_builder=student_optimizer_groups, recovery_state=recovery)
                for name,p in model.named_parameters():
                    if name in expected_named: self.assertTrue(torch.equal(p.grad, expected_named[name]), name)
                self.assertEqual(tensor_state_digest(task_slices(initial,0,(2,1),False)),
                                 tensor_state_digest(task_slices(method_state(model),0,(2,1),False)))
                for key,value in method_state(model).items():
                    if key.startswith(('adapter_bank.task_adapters.1.', 'cross_view_bank.tasks.1.')):
                        self.assertTrue(torch.equal(value, initial[key]), key)
                self.assertTrue(torch.equal(model.selectors[1], initial['selectors'][1]))
                self.assertTrue(torch.equal(model.head.weight[2:], initial_head['weight'][2:]))
                if arm != 'HEAD_TRAINABLE_ATTENTION':
                    self.assertEqual(tensor_state_digest(model.head.state_dict()), tensor_state_digest(initial_head))
                    self.assertFalse(any(n.startswith('head.') for g in recovery['optimizer_parameter_names'] for n in g))
                if arm == 'BASE_FIXED_ATTENTION':
                    names = [n for g in recovery['optimizer_parameter_names'] for n in g]
                    self.assertTrue(names and all(n.startswith('cross_view_bank.tasks.0.') for n in names))
                self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_rng_restored_after_training_and_exceptions_and_shared_across_arms(self):
        before = rng_digest(rng_state()); traces = []
        for arm in ARMS:
            with adaptation_rng(701): traces.append((random.random(), float(np.random.rand()), torch.rand(2)))
            self.assertEqual(before, rng_digest(rng_state()))
        for trace in traces[1:]:
            self.assertEqual(trace[:2], traces[0][:2]); self.assertTrue(torch.equal(trace[2], traces[0][2]))
        with self.assertRaises(RuntimeError):
            with adaptation_rng(8):
                torch.rand(3); raise RuntimeError('interrupted')
        self.assertEqual(before, rng_digest(rng_state()))


if __name__ == '__main__': unittest.main()
