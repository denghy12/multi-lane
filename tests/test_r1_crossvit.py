import copy
from pathlib import Path
import tempfile
import unittest

import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.r1_crossvit import R1Receiver,CLSAgent,EvidenceSource,compact
from multi_lane.track_a.runner import TASK_SIZES,train_task,build_optimizer_groups
from test_track_a_reproduction import FakeVisual


class R1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setup_model(self,kind='native',plain=False,view='full'):
        torch.manual_seed(2026)
        c=MultiLaneModel if plain else R1Receiver
        extra={} if plain else dict(r1_kind=kind,r1_view=view,r1_layer=0,r1_dim=4,r1_heads=2)
        m=c(FakeVisual(),TASK_SIZES,num_selectors=10,num_prompts=2,num_prompt_layers=1,
            adapter_mode='image_token',adapter_bottleneck_dim=4,adapter_layer_indices=(0,),adapter_residual_scale=.1,**extra)
        m.activate_task(0)
        x=torch.randn(4,3,4,4)
        y=torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],[0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]])
        return m,x,y

    def evidence(self,x):
        return dict(full=x,evidence_full=x*.8,evidence_person=x*.7,evidence_face=x*.6,
            evidence_face_reliable=torch.tensor([True,False,True,False],device=x.device))

    def train(self,m,x,y,task=0,epochs=2,amp=False):
        return train_task(m,[(x,y)],[(x,y)],next(m.parameters()).device,task,epochs,.001,0.,1.,amp,
            adapter_learning_rate=.0004,loss_routing='adapter_asl')

    def test_native_forward_and_training_exactly_match_archived_model(self):
        a,x,y=self.setup_model(plain=True);b,xx,yy=self.setup_model()
        self.assertTrue(torch.equal(x,xx))
        self.assertTrue(torch.equal(a.current_all_logits(x),b.current_all_logits(x)))
        ar=self.train(a,x,y);rng=torch.get_rng_state().clone()
        br=self.train(b,xx,yy)
        for n,v in compact(a).items():self.assertTrue(torch.equal(v,compact(b)[n]),n)
        self.assertEqual([r['current_loss'] for r in ar],[r['current_loss'] for r in br])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_new_initialization_preserves_original_rng_and_parameters(self):
        base,x,y=self.setup_model();rng=torch.get_rng_state().clone()
        for kind in ('attention','pool'):
            m,xx,yy=self.setup_model(kind)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(x,xx))
            for n,v in compact(base).items():self.assertTrue(torch.equal(v,compact(m)[n]),n)

    def test_zero_readout_and_off_preserve_receiver_native_forward(self):
        for view in ('full','person','face'):
            for kind in ('attention','pool'):
                m,x,_=self.setup_model(kind,view=view)
                a=m.current_all_logits(self.evidence(x));m.r1_enabled=False
                self.assertTrue(torch.equal(a,m.current_all_logits(x)),(kind,view))

    def test_unreliable_face_cannot_affect_other_receivers(self):
        for kind in ('attention','pool'):
            for view in ('full','person'):
                m,x,_=self.setup_model(kind,view=view)
                with torch.no_grad():m.r1_bank[0].out.weight.normal_(std=.1)
                inputs=self.evidence(x);inputs['evidence_face_reliable'].zero_()
                a=m.current_all_logits(inputs)
                inputs['evidence_face']=torch.randn_like(x)*10
                self.assertTrue(torch.equal(a,m.current_all_logits(inputs)))

    def test_own_supervision_trains_receiver_and_bank_only(self):
        for kind in ('attention','pool'):
            m,x,y=self.setup_model(kind)
            rep,adapter,groups=build_optimizer_groups(m,0.,adapter_learning_rate=.0004)
            ids=[id(p) for g in groups for p in g['params']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(set(ids),{id(p) for p in (*rep,*adapter)})
            inputs=self.evidence(x)
            for k in ('evidence_full','evidence_person','evidence_face'):inputs[k].requires_grad_(True)
            rows=self.train(m,inputs,y)
            self.assertTrue(all(r['skipped_optimizer_steps']==0 for r in rows))
            self.assertGreater(float(m.r1_bank[0].out.weight.norm()),0)
            self.assertTrue(all(inputs[k].grad is None for k in ('evidence_full','evidence_person','evidence_face')))
            m.assert_visual_frozen()

    def test_epoch_recovery_restores_optimizer_scheduler_and_model(self):
        with tempfile.TemporaryDirectory() as folder:
            m,x,y=self.setup_model('attention');m.r1_root=Path(folder)/'first';m.r1_config={'test':1}
            self.train(m,self.evidence(x),y,epochs=3);expected=compact(m)
            clone,xx,yy=self.setup_model('attention');clone.restore_task(-1)
            clone.r1_root=Path(folder)/'second';clone.r1_config={'test':1}
            target=clone.r1_root/'epoch_recovery/task0';target.mkdir(parents=True)
            (target/'epoch01.pth').write_bytes((m.r1_root/'epoch_recovery/task0/epoch01.pth').read_bytes())
            clone.activate_task(0);self.train(clone,self.evidence(xx),yy,epochs=3)
            self.assertTrue(all(torch.equal(v,compact(clone)[n]) for n,v in expected.items()))

    def test_old_task_readout_is_protected(self):
        m,x,y=self.setup_model('attention');self.train(m,self.evidence(x),y)
        before=m.seen_logits(self.evidence(x)).detach()
        m.activate_task(1);self.train(m,self.evidence(x),y[:,:3],task=1)
        self.assertTrue(torch.allclose(before,m.seen_logits(self.evidence(x))[:,:5],atol=1e-6,rtol=0))
        self.assertTrue(all(not p.requires_grad for p in m.r1_bank[0].parameters()))

    def test_probability_and_logit_fusion_are_different(self):
        a=torch.tensor([-8.,2.]);b=torch.tensor([2.,-1.])
        self.assertFalse(torch.equal(.8*a.sigmoid()+.2*b.sigmoid(),(.8*a+.2*b).sigmoid()))

    def test_evidence_source_keeps_original_receiver_augmentation_rng(self):
        class Source:
            sample_ids=['a'];face_reliable=[True]
            def __init__(self,random):self.random=random
            def __len__(self):return 1
            def __getitem__(self,i):return (torch.rand(3) if self.random else torch.ones(3)),torch.ones(2)
        torch.manual_seed(981);a,y=Source(True)[0];state=torch.get_rng_state()
        torch.manual_seed(981);wrapped=EvidenceSource(Source(True),{'person':Source(False),'face':Source(False)},'full')
        x,yy=wrapped[0]
        self.assertTrue(torch.equal(a,x['full']));self.assertTrue(torch.equal(y,yy))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_pool_attention_parameter_budget(self):
        a=sum(p.numel() for p in CLSAgent(768).parameters())
        b=sum(p.numel() for p in CLSAgent(768,pool=True).parameters())
        self.assertLess(abs(a-b)/a,.01)

    @unittest.skipUnless(torch.cuda.is_available(),'real CUDA required')
    def test_actual_cuda_amp_finite(self):
        m,x,y=self.setup_model('attention');m.cuda();x=x.cuda();y=y.cuda()
        rows=self.train(m,self.evidence(x),y,amp=True)
        self.assertTrue(all(r['skipped_optimizer_steps']==0 for r in rows))


if __name__=='__main__':unittest.main()
