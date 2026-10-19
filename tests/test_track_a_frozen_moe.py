from __future__ import annotations
import copy
import unittest
from unittest.mock import patch
import torch
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import tensor_state_digest
from test_track_a_reproduction import FakeVisual


class FrozenMoETest(unittest.TestCase):
    def model(self):
        torch.manual_seed(71)
        visual=FakeVisual()
        visual.transformer.resblocks.extend([copy.deepcopy(visual.transformer.resblocks[0]) for _ in range(2)])
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            view_fusion='fixed_three_view',adapter_mode='moe_frozen_ffn_activate_freeze',
            adapter_layer_indices=(0,),adapter_bottleneck_dim=4,moe_num_experts=5,
            moe_top_k=3,moe_freeze_count=3,moe_freeze_metric='gate_mass',
            moe_noisy_gating=False,moe_dropout=0)

    def images(self):
        return {**{v:torch.randn(3,3,4,4) for v in ('full','person','face')},
                'face_reliable':torch.tensor([True,False,True])}

    def nonzero_experts(self, model):
        with torch.no_grad():
            for expert in model.adapter_bank.experts[0]:expert.up.weight.normal_(0,.07)

    def test_zero_output_matches_original_path_and_initial_rng(self):
        model=self.model();model.activate_task(0);model.eval();images=self.images()
        state=torch.get_rng_state().clone()
        with torch.no_grad():
            enabled=model.seen_logits(images)
            model.adapter_mode='disabled'
            original=model.seen_logits(images)
        self.assertTrue(torch.allclose(enabled,original,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_image_tokens_not_task_tokens_and_live_backward_through_frozen_blocks(self):
        model=self.model();model.activate_task(0);self.nonzero_experts(model);model.train()
        images=self.images();before=tensor_state_digest(model.visual_encoder.state_dict());calls=[]
        original=model.adapter_bank.delta_for_layer
        def record(layer,tokens,lanes,view):
            calls.append((layer,tuple(tokens.shape),list(lanes),view))
            return original(layer,tokens,lanes,view)
        with patch.object(model.adapter_bank,'delta_for_layer',side_effect=record):
            model.current_all_logits_with_views(images)[0].square().sum().backward()
        self.assertEqual(calls,[(0,(1,3,5,8),[0],v) for v in ('full','person','face')])
        bank=model.adapter_bank
        self.assertEqual(bank.training_samples[0,0].tolist(),[3,3,2])
        self.assertTrue(torch.allclose(bank.training_gate_mass[0,0].sum(-1),torch.tensor([3.,3.,2.],dtype=torch.float64)))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in bank.experts.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in bank.tasks[0].parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.visual_encoder.parameters()))
        self.assertEqual(before,tensor_state_digest(model.visual_encoder.state_dict()))

    def test_writeback_reaches_next_summarize_and_router_reads_visual_cls(self):
        model=self.model();model.activate_task(0);self.nonzero_experts(model);model.eval();images=self.images()['full']
        streams=[];gate_inputs=[];base=model._lane_block;route=model.adapter_bank._route
        def record(block,image_tokens,*args,**kwargs):
            streams.append(image_tokens.detach().clone());return base(block,image_tokens,*args,**kwargs)
        def gates(cls,*args,**kwargs):
            gate_inputs.append(cls.detach().clone());return route(cls,*args,**kwargs)
        with torch.no_grad(),patch.object(model,'_lane_block',side_effect=record),patch.object(model.adapter_bank,'_route',side_effect=gates):
            on=model._encode_single_lanes(images,False,'full')
            model.set_adapter_runtime_enabled(False)
            off=model._encode_single_lanes(images,False,'full')
        self.assertTrue(torch.equal(streams[0],streams[3]))
        self.assertFalse(torch.allclose(streams[1],streams[4]))
        self.assertFalse(torch.allclose(on,off))
        b=model.visual_encoder.transformer.resblocks[0];x=streams[0].permute(1,0,2);n=b.ln_1(x)
        expected=(x+b.attn(n,n,n,need_weights=False)[0]).permute(1,0,2)[:,0]
        self.assertTrue(torch.allclose(gate_inputs[0],expected))

    def test_task_conditioned_streams_restore_and_no_future_router(self):
        model=self.model();model.activate_task(0);model.train();images=self.images()
        model.current_all_logits_with_views(images)[0].sum().backward();model.adapter_bank.finish_task(0)
        old=tensor_state_digest(model.adapter_bank.tasks[0].state_dict())
        model.activate_task(1);self.nonzero_experts(model);model.eval();calls=[];original=model.adapter_bank.delta_for_layer
        def record(layer,tokens,lanes,view):
            calls.append((tuple(lanes),view));return original(layer,tokens,lanes,view)
        with torch.no_grad(),patch.object(model.adapter_bank,'delta_for_layer',side_effect=record):a=model.seen_logits(images)
        self.assertEqual(calls,[((t,),v) for v in ('full','person','face') for t in (0,1)])
        self.assertEqual(a.shape,(3,3))
        self.assertEqual(old,tensor_state_digest(model.adapter_bank.tasks[0].state_dict()))
        restored=self.model();restored.load_state_dict(model.state_dict());restored.restore_task(1);restored.eval()
        with torch.no_grad():b=restored.seen_logits(images)
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(any(p.requires_grad for p in restored.visual_encoder.parameters()))

    def test_checkpoint_old_lane_scores_use_own_router_not_current_router(self):
        model=self.model();model.activate_task(0);model.train();images=self.images()
        model.current_all_logits_with_views(images)[0].sum().backward();model.adapter_bank.finish_task(0)
        model.activate_task(1);self.nonzero_experts(model);model.eval()
        with torch.no_grad():
            # Old and new Routers choose distinct subsets of a shared pool.
            model.adapter_bank.tasks[0][0].gates[0].normal_(0,.5)
            a=model.seen_logits(images)
            model.adapter_bank.tasks[1][0].gates[0].normal_(0,9)
            b=model.seen_logits(images)
        self.assertTrue(torch.equal(a[:,:2],b[:,:2]))


if __name__=='__main__':unittest.main()
