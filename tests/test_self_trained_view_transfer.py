import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import test_protected_view_forward as fixtures
from multi_lane.track_a.fusion_function_constraint import FusionFunctionConstraint
from multi_lane.track_a.head_anchored_pilot import student_optimizer_groups, task_slices
from multi_lane.track_a.historical_b32_backbone_attribution import method_state
from multi_lane.track_a.runner import tensor_state_digest, train_task
from multi_lane.track_a.self_trained_view_transfer import compare_probes,configure_phase,initial_core


class SelfTrainedViewTransferTest(unittest.TestCase):
    pair=fixtures.ProtectedViewForwardTest.pair

    def toy(self,include_adapter):
        m=torch.nn.Parameter(torch.tensor(1.)); a=torch.nn.Parameter(torch.tensor(2.))
        teacher=torch.nn.Parameter(torch.tensor(3.)); x=torch.tensor([[1.],[2.]])
        calls=[]
        def predict(images,task):
            calls.append(images is x)
            return torch.cat([teacher*images,1000*teacher*images],dim=1)
        logits=torch.cat([(m+a)*x,m*x],dim=1)
        base_m=(m-4).square();base_a=(a-4).square()
        constraint=FusionFunctionConstraint(predict,[0],include_adapter)
        # Avoid a zero functional residual in this toy.
        with torch.no_grad():teacher.fill_(4.)
        result=constraint(images=x,logits=logits,task_id=0,model_loss=base_m,adapter_loss=base_a,
            model_parameters=[m],adapter_parameters=[a])
        return m,a,teacher,base_m,base_a,constraint,result,calls

    def test_teacher_is_target_only_and_adapter_constraint_is_explicit(self):
        m,a,teacher,bm,ba,c,r,calls=self.toy(False)
        self.assertIsNone(torch.autograd.grad(r['model_penalty'],teacher,
            allow_unused=True,retain_graph=True)[0])
        gm=torch.autograd.grad(bm+r['model_penalty'],m,retain_graph=True)[0]
        ga=torch.autograd.grad(ba+r['adapter_penalty'],a)[0]
        self.assertEqual(calls,[True]);self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(float(ga),-4.)
        self.assertNotEqual(float(gm),-6.)
        self.assertGreater(c.calibration['beta_model'],0)
        self.assertEqual(c.calibration['beta_adapter'],0)

    def test_adapter_also_receives_function_gradient_when_enabled(self):
        m,a,teacher,bm,ba,c,r,calls=self.toy(True)
        ga=torch.autograd.grad(ba+r['adapter_penalty'],a)[0]
        self.assertNotEqual(float(ga),-4.)
        self.assertGreater(c.calibration['beta_adapter'],0)
        self.assertIsNone(teacher.grad)
        self.assertEqual(r['metrics']['teacher_mse'],2.5)  # Class 1 cannot enter D.

    def test_calibration_is_fixed_and_degenerate_gradient_is_safe(self):
        m,a,teacher,bm,ba,c,r,calls=self.toy(True)
        initial=copy.deepcopy(c.calibration)
        x=torch.tensor([[3.]])
        c(images=x,logits=torch.cat([(m+a)*x,m*x],1),task_id=0,
            model_loss=(m-5).square(),adapter_loss=(a-1).square(),model_parameters=[m],adapter_parameters=[a])
        self.assertEqual(c.calibration,initial)
        self.assertEqual(c.coefficient(1.,0.),0.)
        self.assertEqual(c.coefficient(0.,1.),0.)
        self.assertEqual(c.coefficient(1.,1e-6),1.)

    def test_student_keeps_own_head_and_own_old_task_parameters(self):
        model,_=self.pair();head=tensor_state_digest(model.head.state_dict())
        old=tensor_state_digest(task_slices(method_state(model),1,model._task_sizes,False))
        configure_phase(model,1,False)
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        self.assertFalse(model.cross_view_runtime_enabled)
        self.assertTrue(all(not p.requires_grad for p in model.cross_view_bank.parameters()))
        model.head.weight.grad=torch.ones_like(model.head.weight)
        configure_phase(model,1,True)
        self.assertTrue(all(not p.requires_grad for p in model.head.parameters()))
        self.assertIsNone(model.head.weight.grad)
        self.assertTrue(model.cross_view_runtime_enabled)
        self.assertTrue(all(p.requires_grad for p in model.cross_view_bank.tasks[1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.cross_view_bank.tasks[0].parameters()))
        self.assertEqual(head,tensor_state_digest(model.head.state_dict()))
        self.assertEqual(old,tensor_state_digest(task_slices(method_state(model),1,model._task_sizes,False)))
        self.assertEqual(set(initial_core(model)),{'selectors','prompts.0','head.weight','head.bias'})

    def test_runner_hook_changes_only_declared_objectives(self):
        model,_=self.pair();model.restore_task(0);configure_phase(model,0,True)
        loader=DataLoader(TensorDataset(torch.randn(3,3,4,4),torch.ones(3,2)),batch_size=3)
        # Use the same fake visual with single-view mode for a small runner integration check.
        model.view_fusion='disabled';model.view_auxiliary_loss_weight=0.
        calls=[]
        def constraint(**kwargs):
            calls.append(kwargs['task_id'])
            return dict(model_penalty=kwargs['logits'].float().square().mean()*.01,
                adapter_penalty=kwargs['logits'].sum()*0.,metrics=dict(test_marker=7.))
        model.cross_view_runtime_enabled=False
        model.cross_view_bank.requires_grad_(False)
        with patch('multi_lane.track_a.runner.task_indices',return_value=range(2)):
            history=train_task(model,loader,None,torch.device('cpu'),0,1,.001,0.,1.,False,
                adapter_learning_rate=.001,loss_routing='adapter_asl',optimizer_updates_per_task=1,
                optimizer_group_builder=student_optimizer_groups,functional_constraint=constraint)
        self.assertEqual(calls,[0]);self.assertEqual(history[0]['functional_test_marker'],7.)

    def test_probe_distinguishes_roundoff_from_drift_and_checks_negative_control(self):
        before={'fused':torch.tensor([[.1,.2]])}
        repeat={'fused':before['fused']+1e-7}
        after={'fused':before['fused']+2e-7}
        audit=compare_probes(before,repeat,after)
        self.assertTrue(audit['within_tolerance'])
        self.assertFalse(audit['bitwise_prediction_equality_required'])
        with self.assertRaises(AssertionError):compare_probes(before,repeat,{'fused':before['fused']+.001})
        with self.assertRaises(AssertionError):compare_probes(before,{'fused':before['fused']+.001},after)
