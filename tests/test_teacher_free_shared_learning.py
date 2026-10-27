"""Mechanism tests for the teacher-free experiment protocol and recovery."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.teacher_free_shared_learning import (
    AuditedLoader, complementary_options, compare_old_probe, configure_current,
    load_own, parser, smoke_update_budget, task_slices, training_config, validate_target_tasks)
from multi_lane.track_a.run_teacher_free_shared_learning import (
    GATE, INITIAL_ARMS, enqueue_stage, promotion_decision)
from multi_lane.track_a.run_historical_b32_test_attribution import write
from multi_lane.track_a.task_recovery import rng_state, restore_rng


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.selectors=torch.nn.Parameter(torch.ones(3,2))
        self.prompts=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2,3,2))])
        self.adapter_bank=torch.nn.Module()
        self.adapter_bank.task_adapters=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(3)])
        self.head=torch.nn.Linear(2,5)
        self.visual_encoder=torch.nn.Linear(2,2)
        self.cross_view_bank=None
        self.spatial_summary_bank=None
        self.current_task=None

    def restore_task(self,task):
        self.current_task=task

    def assert_visual_frozen(self):
        assert not any(p.requires_grad for p in self.visual_encoder.parameters())


class TeacherFreeProtocolTests(unittest.TestCase):
    def test_arms_keep_baseline_optional_and_isolate_two_mechanisms(self):
        self.assertIsNone(complementary_options('U0'))
        self.assertFalse(complementary_options('U1')['full_conditioned'])
        self.assertEqual(complementary_options('U2')['mode'],'none')
        self.assertEqual(complementary_options('U3')['mode'],'full')
        self.assertEqual(complementary_options('U4')['mode'],'full_fusion')
        self.assertNotIn('diagnostic_force_ramp',complementary_options('U4'))
        self.assertEqual(complementary_options('U4',True)['diagnostic_force_ramp'],1.)
        with self.assertRaises(ValueError):complementary_options('A_B_teacher')

    def test_formal_stages_cannot_silently_change_budget(self):
        self.assertEqual(validate_target_tasks(1),[0])
        self.assertEqual(validate_target_tasks(3),[0,1,2])
        self.assertEqual(validate_target_tasks(8),list(range(8)))
        self.assertEqual(validate_target_tasks(1,True),[0,1])
        for invalid in (0,2,4,9):
            with self.assertRaises(ValueError):validate_target_tasks(invalid)

    def test_head_remains_trainable_and_only_current_adapter_is_enabled(self):
        model=ToyModel()
        configure_current(model,1)
        self.assertEqual(model.current_task,1)
        self.assertTrue(model.head.weight.requires_grad)
        self.assertTrue(model.head.bias.requires_grad)
        self.assertTrue(model.selectors.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.prompts))
        self.assertTrue(all(p.requires_grad for p in model.adapter_bank.task_adapters[1].parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.adapter_bank.task_adapters[0].parameters()))
        self.assertFalse(model.cross_view_runtime_enabled)
        model.cross_view_bank=torch.nn.Linear(2,2)
        with self.assertRaises(ValueError):configure_current(model,1)

    def test_audited_loader_does_not_add_an_iterator_or_change_batches(self):
        data=[({'full':torch.full((2,),float(i)), 'person':torch.full((2,),float(i+1))},
            torch.tensor([i%2,(i+1)%2],dtype=torch.float32),'person'+str(i)) for i in range(9)]
        left=DataLoader(data,batch_size=2,shuffle=True,
            generator=torch.Generator().manual_seed(31))
        right=DataLoader(data,batch_size=2,shuffle=True,
            generator=torch.Generator().manual_seed(31))
        audited=AuditedLoader(left)
        actual=list(audited)
        expected=list(right)
        self.assertEqual(len(actual),5)
        for (x,y),(rx,ry,_) in zip(actual,expected):
            self.assertTrue(torch.equal(x['full'],rx['full']))
            self.assertTrue(torch.equal(y,ry))
        self.assertEqual(len(audited.epochs),1)
        self.assertEqual(len(audited.epochs[0]),2)
        self.assertTrue(torch.equal(left.generator.get_state(),right.generator.get_state()))

    def test_old_task_slice_excludes_current_adapter_and_current_head(self):
        state={'selectors':torch.arange(12).view(3,4),
            'prompts.0':torch.arange(24).view(2,3,4),
            'head.weight':torch.arange(20).view(5,4),
            'head.bias':torch.arange(5),
            'adapter_bank.task_adapters.0.weight':torch.ones(2,2),
            'adapter_bank.task_adapters.1.weight':torch.ones(2,2)*2,
            'visual_encoder.weight':torch.ones(2,2)}
        old=task_slices(state,1,[2,2,1])
        self.assertEqual(old['selectors'].shape,(1,4))
        self.assertEqual(old['prompts.0'].shape,(2,1,4))
        self.assertEqual(old['head.weight'].shape,(2,4))
        self.assertIn('adapter_bank.task_adapters.0.weight',old)
        self.assertNotIn('adapter_bank.task_adapters.1.weight',old)
        self.assertNotIn('visual_encoder.weight',old)

    def test_probe_detects_changed_old_prediction(self):
        before={'full':torch.tensor([1.,2.]),'fused':torch.tensor([1.,3.])}
        self.assertTrue(compare_old_probe(before,copy.deepcopy(before))['checked'])
        after=copy.deepcopy(before);after['full'][0]+=.001
        with self.assertRaises(AssertionError):compare_old_probe(before,after)
        self.assertFalse(compare_old_probe(None,None)['checked'])

    def test_own_recovery_checks_config_and_restores_rng_without_visual_load(self):
        original_rng=rng_state()
        try:
            with tempfile.TemporaryDirectory() as d:
                model=ToyModel()
                state={k:v.clone() for k,v in model.state_dict().items()
                    if not k.startswith('visual_encoder.')}
                expected_rng=rng_state()
                path=Path(d)/'own.pth'
                torch.save(dict(config={'arm':'U4'},model=state,task_id=1,
                    rng=expected_rng,provenance={}),path)
                model.head.weight.data.add_(5.)
                torch.rand(7)
                recovered=load_own(model,path,{'arm':'U4'})
                self.assertEqual(recovered['task_id'],1)
                self.assertTrue(torch.equal(model.head.weight,state['head.weight']))
                self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng['torch_cpu']))
                with self.assertRaises(ValueError):load_own(model,path,{'arm':'U2'})
        finally:restore_rng(original_rng)

    def test_parser_has_no_external_teacher_or_assembly_arguments(self):
        args=parser().parse_args(['train','--arm','U4','--tasks','1','--output','output/test'])
        self.assertEqual(args.tasks,1)
        self.assertEqual(args.epochs,30)
        self.assertFalse(any(k in vars(args) for k in ('person_run','dual_run','teacher_root')))

    def test_smoke_budget_can_measure_24_updates_without_changing_formal_budget(self):
        args=parser().parse_args(['train','--arm','U4','--output','output/test',
            '--smoke','--smoke-updates','24'])
        self.assertEqual(smoke_update_budget(args),24)
        self.assertEqual(smoke_update_budget(SimpleNamespace(smoke=False,smoke_updates=24)),None)
        args.smoke_updates=0
        with self.assertRaises(ValueError):smoke_update_budget(args)
        with self.assertRaises(SystemExit):
            parser().parse_args(['train','--output','output/test','--smoke-updates','0'])

    def test_config_records_diagnostic_budget_only_for_smoke(self):
        from multi_lane.track_a.export_compact_test_scores import _sha256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            clip=root/'clip.pt';clip.write_bytes(b'pretrained')
            write(root/'config.json',{})
            args=SimpleNamespace(clip_checkpoint=clip,arm='U4',smoke=False,
                smoke_updates=24,source_run=root,data_root=root/'data',
                face_manifest_root=root/'face',historical_reference_root=None)
            base={'clip_checkpoint_sha256':_sha256(clip)}
            formal=training_config(args,{'commit':'same'},base,'initial')
            self.assertNotIn('smoke_updates',formal)
            args.smoke=True
            smoke=training_config(args,{'commit':'same'},base,'initial')
            self.assertEqual(smoke['smoke_updates'],24)
            self.assertEqual(smoke['epochs_per_task'],30)


class PromotionGateTests(unittest.TestCase):
    def fixture(self,root):
        for split in ('val','test'):
            for arm,full,fused in (('U0',50.,53.),('U2',50.,53.01),('U4',49.95,53.03)):
                write(root/arm/split/'task0.json',
                    dict(metrics={'mAP':fused},views={'metrics':{'full':{'mAP':full}}}))
        write(root/'U4/train/training_history.json',
            {'0':[{'complementary_accepted_batches':0.},
                  {'complementary_accepted_batches':.25,'complementary_effective_nonzero_update':.25}]})

    def test_only_predeclared_both_split_signal_promotes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            decision=promotion_decision(root)
            self.assertEqual(decision['status'],'promote')
            self.assertFalse(decision['automatic_eight_task_run'])
            self.assertTrue(decision['test_used_for_development'])
            self.assertEqual(decision['next_tasks'],[0,1,2])

    def test_face_improvement_cannot_excuse_full_decline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            write(root/'U4/test/task0.json',
                dict(metrics={'mAP':53.5},views={'metrics':{'full':{'mAP':49.8}}}))
            decision=promotion_decision(root)
            self.assertEqual(decision['status'],'stop_after_task0')
            self.assertTrue(any('Full fell' in x for x in decision['reasons']))

    def test_guard_that_rejects_every_update_cannot_promote(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            write(root/'U4/train/training_history.json',{'0':[{}]})
            decision=promotion_decision(root)
            self.assertEqual(decision['status'],'stop_after_task0')
            self.assertFalse(decision['nonzero_auxiliary_update'])

    def test_fusion_must_beat_unprotected_on_each_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            write(root/'U2/val/task0.json',
                dict(metrics={'mAP':53.04},views={'metrics':{'full':{'mAP':50.}}}))
            decision=promotion_decision(root)
            self.assertEqual(decision['status'],'stop_after_task0')
            self.assertTrue(any('below U2' in x for x in decision['reasons']))

    def test_positive_alpha_without_rounded_parameter_change_cannot_promote(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            write(root/'U4/train/training_history.json',
                {'0':[{'complementary_accepted_batches':.5,'complementary_effective_nonzero_update':0.}]})
            decision=promotion_decision(root)
            self.assertEqual(decision['status'],'stop_after_task0')
            self.assertFalse(decision['effective_nonzero_update'])

    def test_tiny_single_seed_change_is_not_automatic_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            for split in ('val','test'):
                write(root/'U4'/split/'task0.json',
                    dict(metrics={'mAP':53.015},views={'metrics':{'full':{'mAP':50.}}}))
            self.assertEqual(promotion_decision(root)['status'],'stop_after_task0')

    def test_completed_prefix_uses_evaluation_jobs_not_duplicate_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for arm in INITIAL_ARMS:
                write(root/arm/'train/training_complete.json',{'status':'complete','tasks':[0]})
            jobs=enqueue_stage(SimpleNamespace(),root,INITIAL_ARMS,1)
            self.assertEqual(len(jobs),6)
            self.assertTrue(all(job['kind']=='eval' for job in jobs))
            self.assertEqual({job['split'] for job in jobs},{'val','test'})


if __name__=='__main__':
    unittest.main()

