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

    def model(self,layer=8):
        torch.manual_seed(99);v=FakeVisual()
        v.transformer.resblocks.extend(copy.deepcopy(v.transformer.resblocks[0]) for _ in range(11))
        m=FrozenR1Receiver(v,TASK_SIZES,num_selectors=3,num_prompts=2,num_prompt_layers=1,
            adapter_mode='image_token',adapter_bottleneck_dim=4,adapter_layer_indices=(0,),
            r1_view='person',r1_layer=layer,r1_dim=8,r1_heads=2)
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

    def test_fp32_nonfinite_guard_prevents_any_parameter_update(self):
        m,x=self.model();m.r1_require_finite_gradients=True
        before=compact(m)
        handle=m.head.weight.register_hook(lambda g:g*float('nan'))
        y=torch.zeros(4,5);y[:,0]=1
        with self.assertRaises(FloatingPointError):
            train_task(m,[(x,y)],[(x,y)],torch.device('cpu'),0,1,.001,0.,1.,False,
                adapter_learning_rate=.0004,loss_routing='adapter_asl')
        handle.remove()
        self.assertTrue(all(torch.equal(v,compact(m)[n]) for n,v in before.items()))

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
