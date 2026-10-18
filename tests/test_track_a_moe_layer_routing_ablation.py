"""Independent dispatch/freeze controls and single-block placement invariants."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from multi_lane.track_a.moe_activate_freeze import TaskForwardActivateFreezeBank
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import tensor_state_digest
from multi_lane.track_a.verify_moe_activate_freeze import verify
from test_track_a_reproduction import FakeVisual, FakeResidualBlock


class LayerRoutingAblationTest(unittest.TestCase):
    def bank(self, k=2, f=2, experts=4, layer=1):
        bank=TaskForwardActivateFreezeBank(2,8,4,(layer,),.1,experts,
                    top_k=k,freeze_count=f,dropout=0,noisy_gating=False,freeze_metric='gate_mass')
        bank.activate_task(0)
        return bank

    def test_dispatch_and_freeze_counts_are_independent_and_allocation_matched(self):
        allocations=[]
        for k,f in ((2,2),(2,3),(3,2),(3,3)):
            with self.subTest(k=k,f=f):
                bank=self.bank(k,f)
                allocations.append(bank.total_parameter_count())
                x=torch.ones(1,3,2,8,requires_grad=True)
                bank.set_face_reliability(torch.tensor([True,False,True]))
                for view in bank.view_names:bank.delta_for_layer(1,x,[0],view)
                self.assertEqual(bank.training_counts[0,0].sum(1).tolist(),[3*k,3*k,2*k])
                self.assertTrue(torch.allclose(bank.training_gate_mass[0,0].sum(1),torch.tensor([3.,3.,2.],dtype=torch.float64)))
                selected=bank.finish_task(0)['1']['selected_experts']
                self.assertEqual(len(selected),f)
                self.assertEqual(int(bank.frozen_experts.sum()),f)
        self.assertEqual(len(set(allocations)),1)

    def test_freeze_three_reselects_without_topup_and_roundtrips_strictly(self):
        bank=self.bank(2,3)
        bank.training_counts[0,0,0]=torch.tensor([2,2,2,2])
        bank.training_gate_mass[0,0,0]=torch.tensor([.1,.4,.8,.7],dtype=torch.float64)
        record=bank.finish_task(0)['1']
        self.assertEqual(record['selected_experts'],[2,3,1])
        self.assertNotIn('top2',record)
        state=copy.deepcopy(bank.state_dict())
        self.assertIn('freeze_topk',state)
        self.assertNotIn('freeze_top2',state)
        restored=self.bank(2,3);restored.load_state_dict(state,strict=True);restored.restore_task(0)
        restored.activate_task(1)
        restored.training_counts[1,0,0]=bank.training_counts[0,0,0]
        restored.training_gate_mass[1,0,0]=bank.training_gate_mass[0,0,0]
        record=restored.finish_task(1)['1']
        self.assertEqual(record['newly_frozen'],[])
        self.assertEqual(record['remaining_trainable'],1)
        with self.assertRaises(RuntimeError):self.bank(2,2).load_state_dict(state,strict=True)

    def test_default_preserves_legacy_checkpoint_schema(self):
        bank=self.bank()
        self.assertEqual(tuple(bank.state_dict()['freeze_top2'].shape),(2,1,2))
        self.assertNotIn('freeze_topk',bank.state_dict())
        self.assertFalse(any(k.startswith('route_') for k in bank.state_dict()))
        for k in (2,3):
            for f in (2,3):
                other=self.bank(k,f)
                self.assertEqual({n for n,_ in bank.named_parameters()},{n for n,_ in other.named_parameters()})

    def test_all_three_frozen_preserve_new_router_and_input_gradient(self):
        bank=self.bank(3,3,experts=3)
        with torch.no_grad():
            for i,expert in enumerate(bank.experts[0]):
                expert.down.weight.fill_(.1)
                expert.up.weight.fill_(.02*(i+1))
        bank.training_counts[0,0,0].fill_(1)
        bank.training_gate_mass[0,0,0].fill_(1/3)
        bank.finish_task(0);bank.activate_task(1)
        before=tensor_state_digest(bank.experts.state_dict())
        optimizer=torch.optim.AdamW(bank.active_parameters(),lr=.01,weight_decay=.1)
        x=torch.ones(1,2,3,8,requires_grad=True)
        bank.delta_for_layer(1,x,[1],'full').square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0)
        self.assertGreater(float(bank.tasks[1][0].gates[0].grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in bank.experts.parameters()))
        optimizer.step()
        self.assertEqual(before,tensor_state_digest(bank.experts.state_dict()))

    def test_eval_weight_diagnostics_preserve_state_and_reset(self):
        bank=self.bank(3,3)
        with torch.no_grad():bank.tasks[0][0].gates[0][0]=torch.tensor([3.,2.,1.,-10.])
        bank.eval()
        before=tensor_state_digest(bank.state_dict())
        x=torch.ones(1,2,3,8)
        bank.delta_for_layer(1,x,[0],'full')
        row=bank.route_statistics()['0']['1']['full']
        self.assertEqual(sum(row['expert_topk_count']),6)
        self.assertAlmostEqual(sum(row['expert_gate_mass']),2.,places=6)
        weights=row['mean_sorted_gate_weights']
        self.assertEqual(len(weights),3)
        self.assertGreater(weights[0],weights[1]);self.assertGreater(weights[1],weights[2])
        self.assertGreater(weights[2],0)
        self.assertEqual(before,tensor_state_digest(bank.state_dict()))
        bank.reset_route_statistics()
        self.assertEqual(bank.route_statistics()['0']['1']['full']['expert_gate_mass'],[0.]*4)

    def test_model_places_single_pool_at_each_requested_block(self):
        for layer in (1,5,11):
            with self.subTest(layer=layer):
                visual=FakeVisual()
                visual.transformer.resblocks=nn.ModuleList([FakeResidualBlock(8,2) for _ in range(12)])
                model=MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,
                          num_prompt_layers=5,view_fusion='fixed_three_view',
                          adapter_mode='moe_task_ffn_activate_freeze',adapter_layer_indices=(layer,),
                          adapter_bottleneck_dim=4,moe_num_experts=4,moe_top_k=3,
                          moe_freeze_count=3,moe_freeze_metric='gate_mass',moe_noisy_gating=False,moe_dropout=0)
                model.activate_task(0);model.train()
                images={v:torch.randn(3,3,4,4) for v in ('full','person','face')}
                images['face_reliable']=torch.tensor([True,False,True])
                before=tensor_state_digest(visual.state_dict())
                bank=model.adapter_bank
                with patch.object(bank,'delta_for_layer',wraps=bank.delta_for_layer) as call:
                    model.current_all_logits_with_views(images)[0].sum().backward()
                visited=[c.args[0] for c in call.call_args_list if c.args[0] in bank.layer_indices]
                self.assertEqual(visited,[layer]*3)
                self.assertEqual(bank.training_samples[0,0].tolist(),[3,3,2])
                self.assertEqual(len(bank.experts),1)
                self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for p in bank.experts.parameters()))
                self.assertEqual(before,tensor_state_digest(visual.state_dict()))

    def test_invalid_or_unused_controls_fail(self):
        for k,f,e in ((1,2,4),(4,2,4),(3,2,2),(2,0,4),(2,5,4),(2,2.5,4),(2,True,4)):
            with self.subTest(k=k,f=f,e=e),self.assertRaises(ValueError):self.bank(k,f,e)
        with self.assertRaises(ValueError):MultiLaneModel(FakeVisual(),(2,1),moe_freeze_count=3)

    def test_verifier_supports_top3_and_rejects_tampered_freeze_set(self):
        bank=self.bank(3,3)
        bank.set_face_reliability(torch.tensor([True,False]))
        for view in bank.view_names:bank.delta_for_layer(1,torch.ones(1,2,3,8,requires_grad=True),[0],view)
        record=bank.finish_task(0)
        source={'dirty':False,'commit':'fixture'}
        config=dict(git=source,max_tasks=1,adapter_mode='moe_task_ffn_activate_freeze',
                    adapter_layer_indices=[1],moe_num_experts=4,moe_top_k=3,moe_freeze_count=3,
                    moe_freeze_metric='gate_mass',moe_freeze_policy='cumulative_training_gate_mass_top3')
        state={'adapter_bank.'+k:v for k,v in bank.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);train=root/'train';(train/'compact_checkpoints').mkdir(parents=True)
            files={'config':config,'seed_summary':dict(status='complete',config=config,task_metrics=[]),
                   'paired_protocol_audit':dict(frozen_visual_unchanged=True),
                   'training_history':{'0':[dict(optimizer_steps=2,skipped_optimizer_steps=0)]},
                   'activate_freeze_audit':{'tasks':{'0':dict(frozen_experts_and_old_routers_unchanged=True,
                       protected_sha256_before='same',protected_sha256_after='same',freeze=record)}}}
            for name,data in files.items():(train/(name+'.json')).write_text(json.dumps(data))
            checkpoint=train/'compact_checkpoints/task0.pth'
            payload=dict(source_git=source,task_id=0,model=state);torch.save(payload,checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):verify(root,1)
            state['adapter_bank.freeze_topk'][0,0,2]=-1
            torch.save(payload,checkpoint)
            with self.assertRaises(AssertionError):verify(root,1)


if __name__=='__main__':unittest.main()
