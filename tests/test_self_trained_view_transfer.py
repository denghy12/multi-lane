import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

import test_protected_view_forward as fixtures
from multi_lane.track_a.fusion_function_constraint import FusionFunctionConstraint,gradient_norm
from multi_lane.track_a.head_anchored_pilot import student_optimizer_groups, task_slices
from multi_lane.track_a.historical_b32_backbone_attribution import method_state
from multi_lane.track_a.runner import tensor_state_digest, train_task
from multi_lane.track_a.self_trained_view_transfer import compare_probes,configure_phase,initial_core
from multi_lane.track_a.self_trained_view_transfer import prediction_probe
from multi_lane.track_a.self_trained_recovery import compatible_config,import_own_recovery,training_ast


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

    def test_small_half_precision_gradients_use_actual_training_scale(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        loss=(p.half()*1e-5).float()*1e-6
        small=torch.autograd.grad(loss*1024,p,retain_graph=True)[0]
        norm,scale=gradient_norm(loss,[p])
        self.assertEqual(float(small),0.)
        self.assertEqual(scale,65536.)
        self.assertGreater(norm,1e-12)

    def test_calibration_overflow_falls_back_without_changing_training_scaler(self):
        p=torch.nn.Parameter(torch.tensor(1.))
        loss=(p.half()*1000).float()*10
        norm,scale=gradient_norm(loss,[p])
        self.assertEqual(scale,1.)
        self.assertAlmostEqual(norm,10000.)

    def test_probe_uses_math_attention_and_restores_backend_flags(self):
        model,_=self.pair();x=fixtures.ProtectedViewForwardTest.images(self)
        original=model.seen_logits_with_views;calls=[]
        before=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.matmul.allow_tf32)
        def checked(images):
            calls.append((torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.matmul.allow_tf32))
            return original(images)
        with patch.object(model,'seen_logits_with_views',checked):prediction_probe(model,(x,None),1)
        self.assertEqual(calls,[(False,True,False,False)])
        self.assertEqual(before,(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),torch.backends.cuda.matmul.allow_tf32))

    def test_recovery_rejects_training_changes_and_different_arm(self):
        old=dict(git={'commit':'old'},arm='SELF_MAIN_CONSTRAINT',batch_size=64)
        compatible_config(old,dict(old,git={'commit':'new'},resume_origin={}))
        with self.assertRaises(ValueError):compatible_config(old,dict(old,batch_size=128))
        with self.assertRaises(ValueError):compatible_config(old,dict(old,arm='SELF_BASE'))
        self.assertNotEqual(training_ast('def train():\n train_task(lr=1)\n'),
                            training_ast('def train():\n train_task(lr=2)\n'))

    def test_own_complete_state_import_preserves_weights_rng_and_original_files(self):
        from multi_lane.track_a.run_historical_b32_test_attribution import write
        from multi_lane.track_a.export_compact_test_scores import _sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';output=root/'output'
            old=dict(git={'commit':'old'},arm='SELF_BASE',seed=1)
            cfg=dict(old,git={'commit':'new'},resume_origin={'training_root':str(source)})
            write(source/'config.json',old)
            state={'selectors':torch.randn(2,3),'head.weight':torch.randn(3,4)}
            compact=source/'compact_checkpoints/task0.pth';compact.parent.mkdir(parents=True)
            torch.save(dict(model=state,source_git=old['git'],task_id=0),compact)
            recovery=source/'recovery_checkpoints/task0.pth';recovery.parent.mkdir(parents=True)
            rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=[])
            torch.save(dict(model=state,source_git=old['git'],config=old,task_id=0,rng=rng,
                provenance={'history':{},'audits':{}}),recovery)
            sha=_sha256(recovery)
            with patch('multi_lane.track_a.self_trained_recovery.verify_training_code',return_value={'exact':True}):
                audit=import_own_recovery(source,output,cfg)
            saved=torch.load(output/'recovery_checkpoints/task0.pth',weights_only=False)
            self.assertTrue(audit['full_same_student_states_only'])
            self.assertEqual(sha,_sha256(recovery))
            self.assertEqual(saved['config'],cfg)
            self.assertEqual(saved['recovery_origin']['weight_training_git'],old['git'])
            self.assertEqual(tensor_state_digest(saved['model']),tensor_state_digest(state))
            self.assertTrue(torch.equal(saved['rng']['torch_cpu'],rng['torch_cpu']))
