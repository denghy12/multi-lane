import copy
import tempfile
from pathlib import Path
import unittest

import torch

from multi_lane.track_a.crossvit_three_view import (ARMS, VIEWS, CLSReadout,
    IndependentThreeView, compact_state, old_state, optimizer_groups)
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, train_task, tensor_state_digest
from multi_lane.track_a.task_recovery import rng_state, restore_rng
from test_track_a_reproduction import FakeVisual


class CrossViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self,arm='C_M',device='cpu'):
        torch.manual_seed(581)
        base=MultiLaneModel(FakeVisual(),TASK_SIZES,num_selectors=10,num_prompts=2,
            num_prompt_layers=1,view_fusion='fixed_three_view',adapter_mode='image_token',
            adapter_bottleneck_dim=4,adapter_layer_indices=(0,),adapter_residual_scale=.03)
        before=torch.get_rng_state().clone()
        m=IndependentThreeView(base,arm,layer=0,dim=4,heads=2).to(device)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        m.activate_task(0)
        torch.manual_seed(938)
        x={v:torch.randn(4,3,4,4,device=device) for v in VIEWS}
        x['face_reliable']=torch.tensor([True,False,True,False],device=device)
        y=torch.tensor([[1.,0.,0.,1.,0.],[0.,1.,0.,0.,1.],
                        [0.,0.,1.,0.,1.],[1.,0.,1.,0.,0.]],device=device)
        return m,x,y

    def train(self,m,x,y,task=0,epochs=2,**kwargs):
        return train_task(m,[(x,y)],None,next(m.parameters()).device,task,epochs,.001,0.,1.,False,
            adapter_learning_rate=.0004,view_auxiliary_loss_weight=.1,loss_routing='adapter_asl',
            optimizer_group_builder=optimizer_groups,**kwargs)

    def test_independent_storages_and_frozen_backbones(self):
        m,_,_=self.model();m.assert_independent()
        a,b=m.branches['full'],m.branches['person']
        self.assertTrue(torch.equal(a.selectors,b.selectors))
        self.assertNotEqual(a.selectors.data_ptr(),b.selectors.data_ptr())
        self.assertTrue(all(not p.requires_grad for b in m.branches.values() for p in b.visual_encoder.parameters()))

    def test_zero_exchange_exactly_replays_each_native_branch(self):
        for arm in ARMS:
            m,x,_=self.model(arm);m.eval()
            with torch.no_grad():
                f,_=m._features(x,False)
                for v,b in m.branches.items():
                    native=b._encode_single_lanes(x[v],False)
                    self.assertTrue(torch.equal(f[v],native),(arm,v))
                on,_=m.seen_logits_with_views(x)
                m.exchange_enabled=False;off,_=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(on,off))

    def test_all_controls_have_paired_initial_task_parameters(self):
        hashes=[]
        for arm in ARMS:
            m,_,_=self.model(arm)
            hashes.append(tensor_state_digest({n:v for n,v in compact_state(m).items() if n.startswith('branches.')}))
        self.assertEqual(len(set(hashes)),1)

    def test_logit_weights_and_valid_mean_are_exact(self):
        for arm in ARMS:
            m,x,_=self.model(arm)
            with torch.no_grad():z,vs=m.seen_logits_with_views(x)
            w=m.last_fusion_weights()[:,0]
            reference=sum(vs[v]*w[:,i,None] for i,v in enumerate(VIEWS))
            self.assertTrue(torch.equal(z,reference))
            self.assertTrue(torch.allclose(w.sum(-1),torch.ones(4)))
            self.assertEqual(float(w[1,2]),0.)
            self.assertAlmostEqual(float(w[0,0]),.64 if arm.endswith('_W') else 1/3,places=6)

    def test_unreliable_face_cannot_change_full_person_or_fusion(self):
        for arm in ('C_M','P_M'):
            m,x,_=self.model(arm)
            with torch.no_grad():
                for blocks in m.exchange:
                    for module in blocks.values():module.out.weight.normal_(std=.1)
                x['face_reliable'].zero_();a,va=m.seen_logits_with_views(x)
                x['face']=torch.randn_like(x['face'])*10
                b,vb=m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(a,b))
                for v in ('full','person'):self.assertTrue(torch.equal(va[v],vb[v]))

    def test_receiver_loss_does_not_update_source_trainable_parameters(self):
        m,x,_=self.model()
        with torch.no_grad():m.exchange[0]['person'].out.weight.normal_(std=.1)
        _,vs=m.current_all_logits_with_views(x)
        source=tuple(m.branches['full'].base_optimizer_parameters())
        grads=torch.autograd.grad(vs['person'].sum(),source,allow_unused=True,retain_graph=True)
        self.assertTrue(all(g is None for g in grads))
        own=torch.autograd.grad(vs['person'].sum(),m.branches['person'].selectors)[0]
        self.assertGreater(float(own.norm()),0.)

    def test_source_tokens_detached_but_query_and_readout_learn(self):
        read=CLSReadout(8,4,2)
        with torch.no_grad():read.out.weight.normal_()
        c=torch.randn(3,1,8,requires_grad=True)
        sources=[torch.randn(3,4,8,requires_grad=True) for _ in range(2)]
        value=read(c,sources,[torch.ones(3,dtype=torch.bool)]*2)
        value.sum().backward()
        self.assertIsNotNone(c.grad)
        self.assertTrue(all(s.grad is None for s in sources))
        self.assertGreater(float(read.q.weight.grad.norm()),0.)

    def test_optimizer_partition_and_training_all_five_controls(self):
        for arm in ARMS:
            m,x,y=self.model(arm)
            before=tensor_state_digest(compact_state(m))
            main,adapter,groups=optimizer_groups(m)
            self.assertFalse({id(p) for p in main}&{id(p) for p in adapter})
            rows=self.train(m,x,y)
            self.assertNotEqual(before,tensor_state_digest(compact_state(m)))
            self.assertTrue(all(r['skipped_optimizer_steps']==0 for r in rows))
            if m.exchange:self.assertGreater(float(m.exchange[0]['full'].out.weight.norm()),0.)

    def test_old_parameters_and_predictions_preserved_after_next_task(self):
        m,x,y=self.model();self.train(m,x,y)
        m.activate_task(1);old=tensor_state_digest(old_state(m,1))
        with torch.no_grad():before=m.seen_logits(x)[:,:5].clone()
        self.train(m,x,y[:,:3],task=1)
        self.assertEqual(old,tensor_state_digest(old_state(m,1)))
        with torch.no_grad():after=m.seen_logits(x)[:,:5]
        self.assertTrue(torch.allclose(before,after,atol=1e-6,rtol=0))
        self.assertTrue(all(not p.requires_grad for p in m.exchange[0].parameters()))

    def test_face_empty_batch_does_not_decay_adam_momentum(self):
        m,x,y=self.model()
        main,adapter,groups=optimizer_groups(m)
        optimizer=torch.optim.Adam(groups,lr=.001)
        for p in (*main,*adapter):p.grad=torch.ones_like(p)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        face_before=tensor_state_digest(m.branches['face'].state_dict())
        ca_before=tensor_state_digest(m.exchange[0]['face'].state_dict())
        m._face_any=False
        for p in (*main,*adapter):p.grad=torch.zeros_like(p)
        m.adapter_gradient_diagnostics();optimizer.step()
        self.assertEqual(face_before,tensor_state_digest(m.branches['face'].state_dict()))
        self.assertEqual(ca_before,tensor_state_digest(m.exchange[0]['face'].state_dict()))

    def test_paired_on_off_frozen_cache_matches_recomputed_forward(self):
        m,x,_=self.model()
        with torch.no_grad():
            m.exchange[0]['full'].out.weight.normal_(std=.1)
            plain,_=m.seen_logits_with_views(x)
            m._frozen_cache={};cached,_=m.seen_logits_with_views(x)
            self.assertTrue(torch.equal(plain,cached))
            m.exchange_enabled=False;off,_=m.seen_logits_with_views(x)
            m._frozen_cache=None;reference,_=m.seen_logits_with_views(x)
            self.assertTrue(torch.equal(off,reference))

    def test_epoch_recovery_replays_optimizer_scheduler_and_rng(self):
        m,x,y=self.model();snap={}
        def callback(epoch,rows,state):
            if epoch==1:
                snap.update(model=compact_state(m),state=copy.deepcopy(state),rng=rng_state())
        self.train(m,x,y,epochs=2,epoch_callback=callback)
        expected=tensor_state_digest(compact_state(m))
        clone,xx,yy=self.model()
        clone.load_state_dict(snap['model'],strict=False);clone.restore_task(0);restore_rng(snap['rng'])
        self.train(clone,xx,yy,epochs=2,resume_training_state=snap['state'])
        self.assertEqual(expected,tensor_state_digest(compact_state(clone)))

    def test_pool_and_attention_parameter_budgets_match(self):
        a=sum(p.numel() for p in CLSReadout(768).parameters())
        b=sum(p.numel() for p in CLSReadout(768,pool=True).parameters())
        self.assertLess(abs(a-b)/a,.01)

    @unittest.skipUnless(torch.cuda.is_available(),'requires real CUDA AMP')
    def test_real_cuda_amp_finite_updates_and_independence(self):
        m,x,y=self.model(device='cuda')
        rows=train_task(m,[(x,y)],None,torch.device('cuda'),0,2,.001,0.,1.,True,
            adapter_learning_rate=.0004,view_auxiliary_loss_weight=.1,loss_routing='adapter_asl',
            optimizer_group_builder=optimizer_groups)
        self.assertTrue(all(r['skipped_optimizer_steps']==0 for r in rows))
        m.assert_independent()


if __name__=='__main__':unittest.main()
