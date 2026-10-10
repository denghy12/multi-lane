from __future__ import annotations
import copy
import io
import unittest
import tempfile
import json
from pathlib import Path

import torch
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.protected_parax import ProtectedParaXBank, PROTECTED_MODES
from multi_lane.track_a.protected_parax_audit import ProtectedRouteAudit, audit_precision
from test_track_a_reproduction import FakeVisual


class ProtectedParaXTest(unittest.TestCase):
    def bank(self, policy="reuse_old"):
        return ProtectedParaXBank(8, 2, 6, (0,), policy=policy, num_tasks=3,
            task_local_router=True, task_local_projection=True,
            initialization="zero_output", output_scale_mode="fixed",
            residual_scale=1., smooth_ratio_bound=.02)

    def update(self, bank, steps=5):
        optimizer = torch.optim.AdamW(bank.active_parameters(), lr=.02, weight_decay=.2)
        x = torch.randn(4, 2, bank.hidden_dim)
        target = torch.randn_like(x)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            output, _ = bank(0, x)
            (output - target).square().mean().backward()
            optimizer.step()
        return x

    def test_identity_gradient_and_actual_bound(self):
        bank = self.bank()
        bank.activate_task(0)
        x = torch.randn(2, 3, 8)
        y, gate = bank(0, x)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        y.square().mean().backward()
        self.assertGreater(float(bank.task_projections[0].weight.grad.norm()), 0)
        self.update(bank)
        self.assertGreater(float(bank.expert_a[0].grad.norm()), 0)
        self.assertGreater(float(bank.expert_b[0].grad.norm()), 0)
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in bank.active_parameters() if p.grad is not None))
        with torch.no_grad():
            bank.task_projections[0].weight.fill_(1e6)
        y, _ = bank(0, x)
        self.assertLessEqual(float(((y-x).norm(dim=-1)/x.norm(dim=-1)).max()), .020001)
        self.assertTrue(bool((gate[:, 2:] == 0).all()))

    def test_frozen_paths_exact_even_with_weight_decay_and_new_experts(self):
        for policy in PROTECTED_MODES.values():
            with self.subTest(policy=policy):
                bank = self.bank(policy)
                bank.activate_task(0)
                x = self.update(bank)
                bank.seal_task(0)
                y, g = bank(0, x, task_id=0)
                for task in (1, 2):
                    old = {k: v.clone() for k,v in bank.protected_state(task).items()}
                    bank.activate_task(task)
                    new_before = {i: bank.expert_a[i].detach().clone() for i in bank.new_expert_ids(task)}
                    self.update(bank)
                    for k,v in old.items():
                        torch.testing.assert_close(v, bank.protected_state(task)[k], atol=0, rtol=0)
                    for i, value in new_before.items():
                        self.assertFalse(torch.equal(value, bank.expert_a[i]))
                    now, ng = bank(0, x, task_id=0)
                    torch.testing.assert_close(y, now, atol=0, rtol=0)
                    torch.testing.assert_close(g, ng, atol=0, rtol=0)
                    bank.seal_task(task)

    def test_access_sets_and_identical_task0(self):
        outputs=[]
        x=torch.randn(3, 1, 8)
        for policy in PROTECTED_MODES.values():
            torch.manual_seed(83)
            bank=self.bank(policy)
            bank.activate_task(0)
            self.update(bank, 2)
            outputs.append(bank(0,x)[0])
            bank.seal_task(0)
            bank.activate_task(1)
            _, gate=bank(0,x)
            expected={"frozen_pool":[0,1], "fresh_only":[2,3], "reuse_old":[0,1,2,3]}[policy]
            self.assertEqual(bank.expert_access[1].nonzero().flatten().tolist(), expected)
            self.assertTrue(bool((gate[:, ~bank.expert_access[1]] == 0).all()))
            self.assertEqual(bank.new_expert_ids(1), () if policy == "frozen_pool" else (2,3))
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, atol=0, rtol=0)

    def test_seal_required_and_state_roundtrip(self):
        bank=self.bank()
        bank.activate_task(0)
        with self.assertRaises(RuntimeError): bank.activate_task(1)
        self.update(bank)
        bank.seal_task(0)
        bank.activate_task(1)
        self.update(bank)
        bank.seal_task(1)
        buffer=io.BytesIO()
        torch.save(bank.state_dict(), buffer)
        buffer.seek(0)
        other=self.bank()
        other.load_state_dict(torch.load(buffer, weights_only=True))
        other.restore_task(1)
        self.assertFalse(any(p.requires_grad for p in other.parameters()))
        x=torch.randn(2,1,8)
        for task in (0,1):
            torch.testing.assert_close(bank(0,x,task_id=task)[0],other(0,x,task_id=task)[0],atol=0,rtol=0)
        other.activate_task(2)
        with self.assertRaises(ValueError): other(0,x,task_id=3)

    def test_audit_precision_restores_training_flags_on_error(self):
        before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with self.assertRaises(RuntimeError):
            with audit_precision():
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                raise RuntimeError('test')
        self.assertEqual(before,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    def test_task_boundary_audit_and_rng_isolation(self):
        torch.manual_seed(93)
        model=MultiLaneModel(FakeVisual(), task_sizes=(2,1), num_selectors=2,
            num_prompts=2, num_prompt_layers=1, view_fusion="fixed_three_view",
            parax_mode="post_task_protected_reuse", parax_layer_indices=(0,),
            parax_rank=2,parax_num_experts=4,parax_initialization="zero_output",
            parax_output_scale_mode="fixed",parax_residual_scale=1.,parax_smooth_ratio_bound=.02)
        images={v:torch.randn(3,3,4,4) for v in ("full","person","face")}
        images["face_reliable"]=torch.tensor([True,False,True])
        loader=[(images,torch.zeros(3,2),["sample0","sample1","sample2"])]
        with tempfile.TemporaryDirectory() as directory:
            audit=ProtectedRouteAudit(Path(directory),torch.device('cpu'))
            for task in (0,1):
                model.activate_task(task)
                rng=torch.get_rng_state().clone()
                audit.begin(model,loader,task)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.update(model.parax_bank)
                rng=torch.get_rng_state().clone()
                audit.finish(model,task)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            rows=json.loads((Path(directory)/'protected_route_audit.json').read_text())['tasks']
            self.assertTrue(rows['1']['protected_tensors_unchanged'])
            self.assertLess(rows['1']['old_task_anchor_drift']['0']['logit_max_absolute_difference'],1e-6)
            self.assertLess(rows['0']['initial_logit_max_difference'],1e-6)

    def test_model_rng_identity_and_old_logits_after_new_learning(self):
        torch.manual_seed(91)
        visual=FakeVisual()
        common=dict(task_sizes=(2,1,1), num_selectors=2, num_prompts=2,
                    num_prompt_layers=1, view_fusion="fixed_three_view")
        torch.manual_seed(92)
        base=MultiLaneModel(copy.deepcopy(visual), **common)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(92)
        model=MultiLaneModel(copy.deepcopy(visual), **common,
            parax_mode="post_task_protected_reuse", parax_layer_indices=(0,),
            parax_rank=2,parax_num_experts=6,parax_initialization="zero_output",
            parax_output_scale_mode="fixed",parax_residual_scale=1.,parax_smooth_ratio_bound=.02)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        images={v:torch.randn(3,3,4,4) for v in ("full","person","face")}
        images["face_reliable"]=torch.tensor([True,False,True])
        base.activate_task(0); model.activate_task(0)
        torch.testing.assert_close(base.current_all_logits(images),model.current_all_logits(images),atol=1e-6,rtol=1e-6)
        for task in range(3):
            if task: model.activate_task(task)
            optimizer=torch.optim.Adam(model.parax_optimizer_parameters(),lr=.02)
            for _ in range(3):
                optimizer.zero_grad(); model.current_all_logits(images).square().mean().backward();optimizer.step()
            model.parax_bank.seal_task(task)
            if task:
                torch.testing.assert_close(old,model.seen_logits_with_views(images)[0][:,:2],atol=1e-6,rtol=1e-6)
            else: old=model.seen_logits_with_views(images)[0].detach()
        model.assert_visual_frozen()

if __name__ == '__main__': unittest.main()
