import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from multi_lane.track_a.moe_activate_freeze import TaskForwardActivateFreezeBank
from multi_lane.track_a.verify_moe_activate_freeze import verify
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES
from test_track_a_reproduction import FakeVisual


class WeightedFreezeTest(unittest.TestCase):
    def bank(self, metric="gate_mass", dropout=0, noise=False):
        bank=TaskForwardActivateFreezeBank(3,8,4,(0,),.1,4,
                                          dropout=dropout,noisy_gating=noise,freeze_metric=metric)
        bank.activate_task(0)
        return bank

    def fractional_fixture(self, bank, task=0):
        bank.training_counts[task,0,0]=torch.tensor([1,1,1,1])
        bank.training_samples[task,0,0]=2
        if bank.freeze_metric=="gate_mass":
            bank.training_gate_mass[task,0,0]=torch.tensor([.2,.4,.8,.6],dtype=torch.float64)

    def test_fractional_rank_disagrees_with_hard_count_and_does_not_truncate(self):
        hard=self.bank("hard_count");weighted=self.bank()
        self.fractional_fixture(hard);self.fractional_fixture(weighted)
        self.assertEqual(hard.finish_task(0)['0']['top2'],[0,1])
        record=weighted.finish_task(0)['0']
        self.assertEqual(record['top2'],[2,3])
        self.assertEqual(record['freeze_scores'],[.2,.4,.8,.6])
        self.assertEqual(record['counts'],[1,1,1,1])
        self.assertFalse(any('gate_mass' in k for k in hard.state_dict()))

    def test_model_dtype_conversion_preserves_accumulator_precision_and_values(self):
        bank=self.bank()
        bank.training_gate_mass[0,0,0,0]=1.0000000001
        original=bank.training_gate_mass.clone()
        for dtype in (torch.float32,torch.float16,torch.float64,torch.float32):
            bank.to(dtype=dtype)
            self.assertEqual(bank.training_gate_mass.dtype,torch.float64)
            self.assertTrue(torch.equal(original,bank.training_gate_mass))
            self.assertEqual(next(bank.parameters()).dtype,dtype)
        bank.float()
        self.assertTrue(torch.equal(original,bank.training_gate_mass))

    def test_actual_gate_mass_counts_mask_and_eval_isolation(self):
        bank=self.bank()
        with torch.no_grad():
            for router in bank.tasks[0][0].gates:
                router[0]=torch.tensor([math.log(999),0.,-100.,-100.])
        bank.set_face_reliability(torch.tensor([True,False]))
        x=torch.ones(1,2,3,8,requires_grad=True)
        for view in bank.view_names:bank.delta_for_layer(0,x,[0],view)
        self.assertEqual(bank.training_counts[0,0].sum(1).tolist(),[4,4,2])
        self.assertTrue(torch.allclose(bank.training_gate_mass[0,0].sum(1),torch.tensor([2.,2.,1.],dtype=torch.float64)))
        self.assertAlmostEqual(float(bank.training_gate_mass[0,0,0,0]),1.998,places=6)
        self.assertAlmostEqual(float(bank.training_gate_mass[0,0,0,1]),.002,places=6)
        before=bank.training_gate_mass.clone();bank.eval()
        for view in bank.view_names:bank.delta_for_layer(0,x,[0],view)
        bank.train()
        with torch.no_grad():bank.delta_for_layer(0,x,[0],'full')
        self.assertTrue(torch.equal(before,bank.training_gate_mass))

    def test_accumulation_preserves_forward_and_backward_with_noise_and_dropout(self):
        hard=self.bank("hard_count",dropout=.1,noise=True)
        with torch.no_grad():
            for expert in hard.experts[0]:expert.up.weight.normal_(std=.1)
        weighted=self.bank(dropout=.1,noise=True)
        result=weighted.load_state_dict(hard.state_dict(),strict=False)
        self.assertEqual(result.missing_keys,['training_gate_mass'])
        x=torch.randn(1,4,3,8)
        torch.manual_seed(71);a=hard.delta_for_layer(0,x,[0],'full');a.square().mean().backward()
        torch.manual_seed(71);b=weighted.delta_for_layer(0,x,[0],'full');b.square().mean().backward()
        self.assertTrue(torch.equal(a,b))
        for (name,p),(other,q) in zip(hard.named_parameters(),weighted.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is None:self.assertIsNone(q.grad)
            else:self.assertTrue(torch.equal(p.grad,q.grad),name)
        self.assertTrue(torch.equal(hard.training_counts,weighted.training_counts))

    def test_mass_checkpoint_roundtrip_and_repeat_freeze_without_topup(self):
        bank=self.bank();self.fractional_fixture(bank);bank.finish_task(0)
        state=copy.deepcopy(bank.state_dict())
        restored=self.bank();restored.load_state_dict(state);restored.restore_task(0)
        self.assertTrue(torch.equal(bank.training_gate_mass,restored.training_gate_mass))
        restored.activate_task(1);self.fractional_fixture(restored,task=1)
        row=restored.finish_task(1)['0']
        self.assertEqual(row['top2'],[2,3]);self.assertEqual(row['newly_frozen'],[])
        self.assertEqual(row['cumulative_frozen'],[2,3])

    def test_model_wires_metric_and_rejects_silent_unused_option(self):
        model=MultiLaneModel(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,
                             num_prompt_layers=1,view_fusion='fixed_three_view',
                             adapter_mode='moe_task_ffn_activate_freeze',adapter_layer_indices=(0,),
                             adapter_bottleneck_dim=4,moe_num_experts=4,
                             moe_freeze_metric='gate_mass',moe_noisy_gating=False,moe_dropout=0)
        model.activate_task(0)
        images={v:torch.randn(3,3,4,4) for v in ('full','person','face')}
        images['face_reliable']=torch.tensor([1,0,1],dtype=torch.bool)
        model.current_all_logits_with_views(images)[0].sum().backward()
        self.assertTrue(torch.allclose(model.adapter_bank.training_gate_mass[0,0].sum(1),torch.tensor([3.,3.,2.],dtype=torch.float64)))
        with self.assertRaises(ValueError):
            MultiLaneModel(FakeVisual(),TASK_SIZES,moe_freeze_metric='gate_mass')

    def test_checkpoint_verifier_checks_mass_not_hard_ranking_and_rejects_mutation(self):
        bank=self.bank();self.fractional_fixture(bank)
        bank.training_counts[0,0,1].copy_(bank.training_counts[0,0,0])
        bank.training_gate_mass[0,0,1].copy_(bank.training_gate_mass[0,0,0])
        bank.training_samples[0,0,1]=2
        record=bank.finish_task(0)
        source={'dirty':False,'commit':'fixture'}
        config=dict(git=source,max_tasks=1,adapter_mode='moe_task_ffn_activate_freeze',
                    adapter_layer_indices=[0],moe_num_experts=4,moe_freeze_metric='gate_mass',
                    moe_freeze_policy='cumulative_training_gate_mass_top2')
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
            state['adapter_bank.training_gate_mass'][0,0,0,2]+=.01
            torch.save(payload,checkpoint)
            with self.assertRaises(AssertionError):verify(root,1)


if __name__=='__main__':unittest.main()
