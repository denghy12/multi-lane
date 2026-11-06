import copy
import sys
import tempfile
from pathlib import Path
import unittest
import torch

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.r1_frozen_cross_view import FrozenR1Receiver
from multi_lane.track_a.r1_crossvit import compact,rng_state
from multi_lane.track_a.runner import TASK_SIZES,train_task
from scripts.emotic.run_r1_frozen_late_stable import run_graph


class LateStableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self,layer=8, view='person', **kwargs):
        torch.manual_seed(99);v=FakeVisual()
        v.transformer.resblocks.extend(copy.deepcopy(v.transformer.resblocks[0]) for _ in range(11))
        m=FrozenR1Receiver(v,TASK_SIZES,num_selectors=3,num_prompts=2,num_prompt_layers=1,
            adapter_mode='image_token',adapter_bottleneck_dim=4,adapter_layer_indices=(0,),
            r1_view=view,r1_layer=layer,r1_dim=8,r1_heads=2, **kwargs)
        m.activate_task(0)
        x=torch.randn(4,3,4,4)
        inputs=dict(full=x,evidence_full=x*.7,evidence_face=x*.6,
            evidence_face_reliable=torch.tensor([True,False,True,False]))
        return m,inputs

    def test_layer9_injection_reaches_later_summarize_and_zero_replays_native(self):
        m,x=self.model();on=m.current_all_logits(x);m.r1_enabled=False
        self.assertTrue(torch.equal(on,m.current_all_logits(x)))
        m.r1_enabled=True
        with torch.no_grad():m.r1_bank[0].out.weight.normal_(std=.1)
        m.current_all_logits(x).square().mean().backward()
        self.assertGreater(float(m.r1_bank[0].out.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in m.visual_encoder.parameters()))

    def test_two_patch_sites_share_only_when_requested_and_second_reads_updated_stream(self):
        for sharing in ('shared', 'independent'):
            with self.subTest(sharing=sharing):
                model, images = self.model(7, frozen_cross_mode='patch',
                    frozen_cross_second_layer=8, frozen_cross_sharing=sharing)
                early = model.cross_agent(0, 7)
                late = model.cross_agent(0, 8)
                self.assertEqual(early is late, sharing == 'shared')
                self.assertEqual((early.cap, late.cap), (.025, .025))
                model.r1_enabled = False
                native = model.current_all_logits(images)
                model.r1_enabled = True
                torch.testing.assert_close(model.current_all_logits(images), native)
                with torch.no_grad():
                    early.out.weight.normal_(std=.2)
                queries = []
                hook = late.register_forward_pre_hook(lambda _, args: queries.append(args[0].detach().clone()))
                model.current_all_logits(images)
                full_query = queries[-1]
                queries.clear()
                model.frozen_cross_active_layers = (8,)
                model.current_all_logits(images)
                self.assertEqual(len(queries), 1)
                self.assertFalse(torch.equal(queries[0], full_query))
                hook.remove()
                model.frozen_cross_active_layers = (7, 8)
                model.current_all_logits(images).square().mean().backward()
                self.assertGreater(float(early.out.weight.grad.norm()), 0)
                self.assertGreater(float(late.out.weight.grad.norm()), 0)
                model.activate_task(1)
                self.assertFalse(any(p.requires_grad for p in model.r1_bank[0].parameters()))

    def test_task_pool_changes_task_state_without_rewriting_patch_query(self):
        model, images = self.model(8, view='face', frozen_cross_mode='patch', task_pool_layer=5)
        images['evidence_person'] = images.pop('evidence_face')
        patch, pool = model.r1_bank[0]
        self.assertEqual(patch.cap, .05)
        with torch.no_grad():
            for module in (patch, pool):
                module.out.weight.normal_(std=.2)
        seen = []
        pool_hook = pool.register_forward_pre_hook(lambda _, args: seen.append(('pool', args[0].detach())))
        patch_hook = patch.register_forward_pre_hook(lambda _, args: seen.append(('patch', args[0].detach())))
        both = model.current_all_logits(images)
        self.assertEqual([name for name, _ in seen], ['pool', 'patch'])
        both_patch_input = seen[-1][1]
        seen.clear()
        model.task_pool_enabled = False
        patch_only = model.current_all_logits(images)
        self.assertEqual([name for name, _ in seen], ['patch'])
        torch.testing.assert_close(both_patch_input, seen[0][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(both, patch_only))
        model.task_pool_enabled = True
        model.frozen_cross_active_layers = ()
        seen.clear()
        model.current_all_logits(images)
        self.assertEqual([name for name, _ in seen], ['pool'])
        pool_hook.remove()
        patch_hook.remove()
        model.frozen_cross_active_layers = (8,)
        both.square().mean().backward()
        self.assertGreater(float(patch.out.weight.grad.norm()), 0)
        self.assertGreater(float(pool.out.weight.grad.norm()), 0)

    def test_external_state_accepts_only_numerical_metadata_changes(self):
        m,_=self.model();m.r1_config={'recipe':'a','git':2,'r1_fixed_loss_scale':4096}
        payload=dict(task=0,config={'recipe':'a','git':1},history=[{'skipped_optimizer_steps':0}],
            model=compact(m),rng=rng_state())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'boundary.pth';torch.save(payload,p)
            m._load_external(p,0)
            m.r1_config['recipe']='different'
            with self.assertRaises(ValueError):m._load_external(p,0)
            m.r1_config['recipe']='a';payload['history'][0]['skipped_optimizer_steps']=1;torch.save(payload,p)
            with self.assertRaises(ValueError):m._load_external(p,0)

    def test_failed_job_does_not_block_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);flag=root/'evaluated'
            jobs=[dict(name='failed',command=[sys.executable,'-c','raise SystemExit(1)'],memory_mib=0),
                dict(name='good',command=[sys.executable,'-c','pass'],memory_mib=0),
                dict(name='blocked',depends=['failed'],command=[sys.executable,'-c','pass'],memory_mib=0),
                dict(name='evaluation',depends=['good'],command=[sys.executable,'-c',f'from pathlib import Path;Path({str(flag)!r}).touch()'],memory_mib=0)]
            result=run_graph(jobs,root,root,available=lambda:100000,poll=.05)
            self.assertEqual(result['failed'],'failed');self.assertEqual(result['blocked'],'blocked')
            self.assertTrue(flag.exists());self.assertEqual(result['evaluation'],'complete')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed')
    def test_fixed_scaler_is_saved_and_has_no_growth(self):
        m,x=self.model();m.cuda();x={k:v.cuda() for k,v in x.items()}
        y=torch.zeros(4,5,device='cuda');y[:,0]=1
        m.r1_fixed_loss_scale=4096
        with tempfile.TemporaryDirectory() as d:
            m.r1_root=Path(d);m.r1_config={'fixed':4096}
            rows=train_task(m,[(x,y)],[(x,y)],torch.device('cuda'),0,2,.001,0,1,True,
                adapter_learning_rate=.0004,loss_routing='adapter_asl')
            self.assertTrue(all(r['skipped_optimizer_steps']==0 for r in rows))
            saved=torch.load(Path(d)/'epoch_recovery/task0/epoch02.pth',map_location='cpu',weights_only=False)
            self.assertEqual(saved['training']['scaler']['scale'],4096)
            self.assertEqual(saved['training']['scaler']['growth_interval'],1000000000)


if __name__=='__main__':unittest.main()
