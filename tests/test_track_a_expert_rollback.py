import copy
import unittest

import torch

from multi_lane.track_a.expert_rollback import (SupportCapture, differences,
                                               old_coordinate_digest, rollback_experts)
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, compact_model_state_dict, tensor_state_digest
from test_track_a_reproduction import FakeVisual


class ExpertRollbackTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(21)
        model = MultiLaneModel(FakeVisual(), TASK_SIZES, num_selectors=2, num_prompts=2,
                              num_prompt_layers=1, view_fusion='fixed_three_view',
                              adapter_mode='moe_task_ffn_activate_freeze', adapter_layer_indices=(0,),
                              adapter_bottleneck_dim=4, moe_num_experts=4,
                              moe_dropout=0, moe_noisy_gating=False)
        model.activate_task(0)
        with torch.no_grad():
            for gate in model.adapter_bank.tasks[0][0].gates:
                gate[:, 2] = 3
                gate[:, 3] = 2
            for expert in model.adapter_bank.experts[0]:
                expert.up.weight.normal_(std=.1)
        model.eval()
        return model

    def test_target_expert_only_and_exception_restores_model(self):
        model=self.model();reference=copy.deepcopy(compact_model_state_dict(model))
        with torch.no_grad():model.adapter_bank.experts[0][2].up.bias.add_(.2)
        current=copy.deepcopy(compact_model_state_dict(model));sha=tensor_state_digest(current)
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with rollback_experts(model,reference,[(0,2)]) as keys:
                self.assertEqual(len(keys),4)
                for k,v in compact_model_state_dict(model).items():
                    self.assertTrue(torch.equal(v,reference[k] if k in keys else current[k]),k)
                raise RuntimeError('sentinel')
        self.assertEqual(tensor_state_digest(compact_model_state_dict(model)),sha)

    def test_accessed_expert_rollback_restores_function_negative_control_does_not(self):
        model=self.model();reference=copy.deepcopy(compact_model_state_dict(model))
        images={v:torch.randn(3,3,4,4) for v in ('full','person','face')}
        images['face_reliable']=torch.tensor([1,0,1],dtype=torch.bool)
        with torch.no_grad():
            before=model.seen_logits(images)
            model.adapter_bank.experts[0][2].up.bias.add_(torch.linspace(-1,1,8))
            after=model.seen_logits(images)
        self.assertGreater(float((before-after).abs().max()),1e-5)
        with rollback_experts(model,reference,[(0,0),(0,1)]):
            with torch.no_grad():negative=model.seen_logits(images)
        self.assertTrue(torch.equal(negative,after))
        with rollback_experts(model,reference,[(0,2)]):
            with torch.no_grad():restored=model.seen_logits(images)
        self.assertTrue(torch.equal(restored,before))

    def test_support_excludes_invalid_face_and_covers_all_full_samples(self):
        model=self.model();bank=model.adapter_bank
        bank._route=lambda cls,pool,view:cls
        capture=SupportCapture(bank,torch.tensor([True,False]))
        gates=torch.tensor([[1.,0,0,0],[0.,0,0,1.]])
        capture(gates,bank.tasks[0][0],2)
        self.assertEqual(capture.pairs,{(0,0)})
        capture(gates,bank.tasks[0][0],0)
        self.assertEqual(capture.pairs,{(0,0),(0,3)})

    def test_old_coordinate_digest_ignores_new_task_but_detects_old_changes(self):
        model=self.model();before=copy.deepcopy(compact_model_state_dict(model))
        after=copy.deepcopy(before)
        after['selectors'][1].add_(1)
        after['head.weight'][5:].add_(1)
        after['prompts.0'][:,1].add_(1)
        after['adapter_bank.experts.0.2.up.weight'].add_(1)
        self.assertEqual(old_coordinate_digest(before,1),old_coordinate_digest(after,1))
        after['prompts.0'][:,0].add_(1)
        self.assertNotEqual(old_coordinate_digest(before,1),old_coordinate_digest(after,1))

    def test_face_difference_uses_only_reliable_people(self):
        a={'logit_face':torch.tensor([[0.],[100.]])};b={'logit_face':torch.zeros(2,1)}
        self.assertEqual(differences(a,b,torch.tensor([True,False]))['logit_face']['max_abs'],0)
        self.assertEqual(differences(a,b)['logit_face']['max_abs'],100)

    def test_invalid_expert_never_modifies_model(self):
        model=self.model();reference=compact_model_state_dict(model)
        sha=tensor_state_digest(reference)
        with self.assertRaises(ValueError):
            with rollback_experts(model,reference,[(0,100)]):pass
        self.assertEqual(tensor_state_digest(compact_model_state_dict(model)),sha)


if __name__=='__main__':unittest.main()
