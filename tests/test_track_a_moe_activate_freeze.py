import copy
import unittest

import torch

from multi_lane.track_a.moe_activate_freeze import TaskForwardActivateFreezeBank
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest
from test_track_a_reproduction import FakeVisual


class ActivateFreezeTest(unittest.TestCase):
    def bank(self, experts=4):
        bank = TaskForwardActivateFreezeBank(3, 8, 4, (0,), .1, experts, dropout=0, noisy_gating=False)
        bank.activate_task(0)
        return bank

    def trained_experts(self, bank):
        with torch.no_grad():
            for i, expert in enumerate(bank.experts[0]):
                expert.down.weight.fill_(.1)
                expert.up.weight.fill_(.02*(i+1))

    def finish0(self, bank):
        bank.training_counts[0,0,0] = torch.tensor([100,1,90,2])
        return bank.finish_task(0)

    def test_cumulative_ranks_all_experts_and_does_not_top_up(self):
        bank=self.bank();record=self.finish0(bank)
        self.assertEqual(record['0']['top2'],[0,2])
        bank.activate_task(1)
        bank.training_counts[1,0,0]=torch.tensor([80,2,70,1])
        record=bank.finish_task(1)
        self.assertEqual(record['0']['top2'],[0,2])
        self.assertEqual(record['0']['newly_frozen'],[])
        self.assertEqual(record['0']['remaining_trainable'],2)
        with self.assertRaises(RuntimeError):bank.finish_task(1)

    def test_train_counts_hard_dispatch_reliable_face_only_and_no_eval_contamination(self):
        bank=self.bank();bank.set_face_reliability(torch.tensor([1,0,1],dtype=torch.bool))
        x=torch.ones(1,3,2,8,requires_grad=True)
        for view in bank.view_names:bank.delta_for_layer(0,x,[0],view)
        self.assertEqual(bank.training_samples[0,0].tolist(),[3,3,2])
        self.assertEqual(bank.training_counts[0,0].sum(1).tolist(),[6,6,4])
        before=bank.training_counts.clone();bank.eval()
        for view in bank.view_names:bank.delta_for_layer(0,x,[0],view)
        self.assertTrue(torch.equal(before,bank.training_counts))
        bank.train()
        with torch.no_grad():bank.delta_for_layer(0,x,[0],'full')
        self.assertTrue(torch.equal(before,bank.training_counts))
        self.assertEqual(sum(bank.finish_task(0)['0']['counts']),16)

    def test_frozen_and_trainable_reuse_optimizer_and_input_gradients(self):
        bank=self.bank();self.trained_experts(bank);self.finish0(bank);bank.activate_task(1)
        with torch.no_grad():bank.tasks[1][0].gates[0][0]=torch.tensor([4.,3.,-2.,-3.])
        frozen_before=tensor_state_digest(bank.experts[0][0].state_dict())
        router_before=tensor_state_digest(bank.tasks[0].state_dict())
        train_before=tensor_state_digest(bank.experts[0][1].state_dict())
        params=list(bank.active_parameters());ids={id(p) for p in params}
        self.assertTrue(all(id(p) not in ids for p in bank.experts[0][0].parameters()))
        optimizer=torch.optim.AdamW(params,lr=.01,weight_decay=.1)
        x=torch.ones(1,2,3,8,requires_grad=True)
        value=bank.delta_for_layer(0,x,[1],'full');value.square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0)
        self.assertGreater(float(bank.tasks[1][0].gates[0].grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in bank.experts[0][0].parameters()))
        optimizer.step()
        self.assertEqual(frozen_before,tensor_state_digest(bank.experts[0][0].state_dict()))
        self.assertEqual(router_before,tensor_state_digest(bank.tasks[0].state_dict()))
        self.assertNotEqual(train_before,tensor_state_digest(bank.experts[0][1].state_dict()))

    def test_router_learns_when_both_selected_experts_are_frozen(self):
        bank=self.bank();self.trained_experts(bank);self.finish0(bank);bank.activate_task(1)
        with torch.no_grad():bank.tasks[1][0].gates[0][0]=torch.tensor([4.,-3.,3.,-4.])
        x=torch.ones(1,2,3,8,requires_grad=True)
        bank.delta_for_layer(0,x,[1],'full').square().mean().backward()
        self.assertGreater(float(bank.tasks[1][0].gates[0].grad.norm()),0)
        self.assertTrue(all(p.grad is None for i in (0,2) for p in bank.experts[0][i].parameters()))
        self.assertTrue(all(p.grad is None for i in (1,3) for p in bank.experts[0][i].parameters()))

    def test_full_frozen_pool_still_has_active_router(self):
        bank=self.bank(experts=2);self.trained_experts(bank)
        bank.training_counts[0,0,0]=torch.tensor([2,2]);bank.finish_task(0);bank.activate_task(1)
        self.assertEqual(len(list(bank.active_parameters())),6)
        x=torch.ones(1,2,3,8,requires_grad=True)
        bank.delta_for_layer(0,x,[1],'full').square().mean().backward()
        self.assertGreater(float(bank.tasks[1][0].gates[0].grad.norm()),0)

    def test_checkpoint_roundtrip_persists_policy_state_not_eval_counters(self):
        bank=self.bank();self.trained_experts(bank);self.finish0(bank);bank.eval()
        x=torch.randn(1,3,2,8)
        expected=bank.delta_for_layer(0,x,[0],'person')
        state=copy.deepcopy(bank.state_dict())
        self.assertFalse(any('route_counts' in k for k in state))
        restored=self.bank();restored.load_state_dict(state);restored.restore_task(0);restored.eval()
        self.assertTrue(torch.equal(expected,restored.delta_for_layer(0,x,[0],'person')))
        self.assertTrue(torch.equal(bank.frozen_experts,restored.frozen_experts))
        restored.activate_task(1)
        self.assertFalse(any(p.requires_grad for p in restored.experts[0][0].parameters()))
        self.assertTrue(all(p.requires_grad for p in restored.experts[0][1].parameters()))
        with self.assertRaises(ValueError):restored.delta_for_layer(0,x,[2],'full')

    def test_sequential_finalization_and_empty_counts_fail(self):
        bank=self.bank()
        with self.assertRaises(RuntimeError):bank.activate_task(1)
        with self.assertRaises(RuntimeError):bank.finish_task(0)

    def test_model_interface_counts_once_and_preserves_frozen_visual(self):
        model=MultiLaneModel(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,
                             num_prompt_layers=1,view_fusion='fixed_three_view',
                             adapter_mode='moe_task_ffn_activate_freeze',adapter_layer_indices=(0,),
                             adapter_bottleneck_dim=4,moe_num_experts=4,moe_noisy_gating=False,moe_dropout=0)
        model.activate_task(0);model.train()
        images={v:torch.randn(3,3,4,4) for v in ('full','person','face')}
        images['face_reliable']=torch.tensor([True,False,True])
        before=tensor_state_digest(model.visual_encoder.state_dict())
        model.current_all_logits_with_views(images)[0].sum().backward()
        self.assertEqual(model.adapter_bank.training_samples[0,0].tolist(),[3,3,2])
        self.assertEqual(before,tensor_state_digest(model.visual_encoder.state_dict()))
        self.assertEqual(sum(model.adapter_bank.per_task_parameter_counts()),model.adapter_bank.total_parameter_count())


if __name__=='__main__':unittest.main()
