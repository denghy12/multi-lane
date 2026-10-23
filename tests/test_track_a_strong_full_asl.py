import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_track_a_reproduction as fixtures
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import (build_optimizer_groups, backward_routed_training_losses,
    compute_training_loss,compute_asymmetric_training_loss,parse_args,strong_full_gradient_diagnostics)
from multi_lane.track_a.strong_full_asl import ARMS,modes,validate_arguments
from multi_lane.track_a.run_strong_full_asl import training_arguments,run_queue
from pathlib import Path
import tempfile


class StrongFullASLTest(unittest.TestCase):
    def model(self,arm='T_ASL_BOTH'):
        torch.manual_seed(71);visual=fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        options=modes(arm);options.pop('loss_routing')
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            strong_full_asl=True,adapter_bottleneck_dim=3,adapter_layer_indices=(0,),
            cross_view_layer=1,cross_view_dim=4,cross_view_heads=2,**options)
    def images(self):
        return dict(full=torch.randn(3,3,4,4),person=torch.randn(3,3,4,4),face=torch.randn(3,3,4,4),
            face_reliable=torch.tensor([True,False,True]),spatial_bbox=torch.tensor([[0.,0.,.5,1.]]*3),spatial_visibility=torch.ones(3))
    def test_all_declared_arguments_validate_without_relaxing_old_profile(self):
        a=SimpleNamespace(data_root=Path('data'),clip_checkpoint=Path('clip'),face_manifest_root=Path('face'),smoke=True)
        for arm in ARMS:
            flags=training_arguments(a,Path('output/demo'),arm)
            self.assertEqual('--face-manifest-root' in flags,arm.startswith('T_'))
            args=parse_args(flags);validate_arguments(args)
            args.view_auxiliary_loss_weight=.1
            with self.assertRaises(ValueError):validate_arguments(args)
    def test_same_view_initial_outputs_and_rng_paired(self):
        models={a:self.model(a) for a in ARMS};images=self.images()
        for m in models.values():m.activate_task(0)
        for a in ['S_BCE','S_ASL']:
            self.assertTrue(torch.equal(models[a].current_logits(images['full']),models['S_BCE'].current_logits(images['full'])))
        for a in ARMS[2:]:self.assertTrue(torch.equal(models[a].current_logits(images),models['T_BCE'].current_logits(images)))
        # The same Full feature initializes identically; fused output need not.
        for m in models.values():
            _,_,groups=build_optimizer_groups(m,0,adapter_learning_rate=.0004)
            ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.optimizer_parameters()})
    def test_mixed_loss_matches_explicit_reference_for_every_parameter(self):
        model=self.model();model.activate_task(0)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n.startswith(('adapter_bank.','cross_view_bank.','spatial_summary_bank.')):p.normal_(0,.15)
        main,adapter,_=build_optimizer_groups(model,0,adapter_learning_rate=.0004)
        self.assertFalse({id(p) for p in main}&{id(p) for p in adapter})
        logits=model.current_all_logits(self.images());targets=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        bce=compute_training_loss(logits,targets,[0,1],1.,'legacy_full_zero')
        asl=compute_asymmetric_training_loss(logits,targets,[0,1],1.,'legacy_full_zero',9.8,0.,.05,1e-8)
        expected_main=torch.autograd.grad(bce,main,retain_graph=True)
        expected_adapter=torch.autograd.grad(asl,adapter,retain_graph=True)
        wrong_adapter=torch.autograd.grad(bce,adapter,retain_graph=True)
        self.assertTrue(any(not torch.allclose(x,y) for x,y in zip(expected_adapter,wrong_adapter)))
        backward_routed_training_losses(bce,asl,main,adapter,torch.cuda.amp.GradScaler(enabled=False),False)
        for p,g in zip(main,expected_main):self.assertTrue(torch.equal(p.grad,g))
        for p,g in zip(adapter,expected_adapter):self.assertTrue(torch.equal(p.grad,g))
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
        before=[p.grad.clone() for p in (*main,*adapter)]
        norms=strong_full_gradient_diagnostics(model,1.)
        self.assertEqual(set(norms),{'strong_grad_'+n+'_norm' for n in ['selectors','prompts','classifier','adapter','attention','position']})
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip((*main,*adapter),before)))
    def test_one_forward_evaluation_matches_previous_fused_metrics_and_scores(self):
        import numpy as np
        from dataclasses import asdict
        from multi_lane.track_a.runner import evaluate
        from multi_lane.track_a.strong_full_evaluation import evaluate_fused_and_views
        model=self.model();model.activate_task(0);model.eval()
        images=self.images();labels=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        loader=[(images,labels,['image0#person=0','image1#person=0','image1#person=1'])]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=evaluate(model,loader,torch.device('cpu'),0,.5,False,root/'old.npz')
            new,_=evaluate_fused_and_views(model,loader,torch.device('cpu'),0,.5,False,root/'new.npz',root/'views.npz')
            self.assertEqual(asdict(original),asdict(new))
            with np.load(root/'old.npz') as a,np.load(root/'new.npz') as b:
                for k in ['logits','probabilities','targets','sample_ids']:self.assertTrue(np.array_equal(a[k],b[k]),k)
    def test_queue_fans_out_dependent_jobs_and_preserves_failed_pending(self):
        class Process:
            pid=9
            def poll(self):return 0
        with tempfile.TemporaryDirectory() as tmp:
            extra=dict(name='eval',module='x',arguments=[])
            with patch('multi_lane.track_a.run_strong_full_asl.memory_state',return_value={0:dict(free=24000,utilization=0)}), \
                 patch('multi_lane.track_a.run_strong_full_asl.subprocess.Popen',side_effect=lambda *x,**y:Process()), \
                 patch('multi_lane.track_a.run_strong_full_asl.time.sleep'):
                rows=run_queue([dict(name='base/train',module='x',arguments=[])],Path(tmp),Path(tmp)/'ok.json',[0],
                    on_success=lambda r:[extra] if r['name'].endswith('/train') else [])
            self.assertEqual([r['name'] for r in rows],['base/train','eval'])
            class Failed:
                pid=10
                def poll(self):return 1
            with patch('multi_lane.track_a.run_strong_full_asl.memory_state',return_value={0:dict(free=24000,utilization=0)}), \
                 patch('multi_lane.track_a.run_strong_full_asl.subprocess.Popen',side_effect=lambda *x,**y:Failed()), \
                 patch('multi_lane.track_a.run_strong_full_asl.time.sleep'):
                with self.assertRaises(RuntimeError):run_queue([dict(name='bad',module='x',arguments=[]),dict(name='pending',module='x',arguments=[])],
                    Path(tmp),Path(tmp)/'failed.json',[0])
            import json
            self.assertEqual(json.loads((Path(tmp)/'failed.json').read_text())['pending'][0]['name'],'pending')
    def test_gpu_queue_spreads_jobs_and_preserves_pending_on_failure(self):
        class Process:
            pid=7
            def poll(self):return 0
        with tempfile.TemporaryDirectory() as tmp:
            jobs=[dict(name='job'+str(i),module='x',arguments=[]) for i in range(2)]
            with patch('multi_lane.track_a.run_strong_full_asl.memory_state',return_value={0:dict(free=24000,utilization=0),2:dict(free=24000,utilization=0)}), \
                 patch('multi_lane.track_a.run_strong_full_asl.subprocess.Popen',side_effect=lambda *x,**y:Process()), \
                 patch('multi_lane.track_a.run_strong_full_asl.time.sleep'):
                rows=run_queue(jobs,Path(tmp),Path(tmp)/'status.json',[0,2])
            self.assertEqual(len(rows),2)

if __name__=='__main__':unittest.main()
