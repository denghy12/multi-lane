import copy
import unittest

import torch

from multi_lane.track_a.protected_joint_attention import ProtectedJointModel, bounded_delta
from multi_lane.track_a.protected_attention_gate_attribution import (
    train_constants, ReadoutCapture, candidate_deltas, MODES)
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest
from test_track_a_reproduction import FakeVisual


class ProtectedGateTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(1028)
        m=ProtectedJointModel(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,
            num_prompt_layers=1,view_fusion='fixed_three_view',adapter_mode='image_token',
            adapter_bottleneck_dim=4,adapter_layer_indices=(0,),adapter_residual_scale=.03)
        m.initialize_interaction('C4',layer=0,dim=4,heads=2);m.activate_task(0);m.requires_grad_(False).eval()
        for unit in m.interaction_bank:
            unit.out.weight.normal_(0,.2)
            unit.gate[0].weight.zero_();unit.gate[0].bias.zero_()
            unit.gate[0].weight[:,12]=1.
            unit.gate[-1].weight.fill_(.2);unit.gate[-1].bias.zero_()
        x={v:torch.randn(4,3,4,4) for v in ('full','person','face')}
        x.update(face_reliable=torch.tensor([True,False,True,False]),
            spatial_bbox=torch.tensor([[0.,0.,.5,1.]]*4),spatial_visibility=torch.ones(4))
        constants=dict(scale=[.15]*8,gate_mean=[.4]*8)
        return m,x,constants

    def test_train_constants_are_only_last_five_and_task_local(self):
        a={str(t):[dict(interaction_delta_to_base_rms=.01)]*30 for t in range(8)}
        b={str(t):[dict(interaction_delta_to_base_rms=.05,interaction_gate_mean=.8)]*30 for t in range(8)}
        b['0'][0]=dict(interaction_delta_to_base_rms=100,interaction_gate_mean=.01)
        c=train_constants(a,b)
        for v in c['scale']:self.assertAlmostEqual(v,.2)
        for v in c['gate_mean']:self.assertAlmostEqual(v,.8)
        invalid=copy.deepcopy(b);invalid['1']=invalid['1'][:-1]
        with self.assertRaises(ValueError):train_constants(a,invalid)

    def test_native_components_replay_and_hooks_are_removed(self):
        m,x,c=self.make();before=tensor_state_digest(m.state_dict())
        with ReadoutCapture(m) as capture,torch.no_grad():
            z,_=m.seen_logits_with_views(x)
            d,g,ratios=candidate_deltas(m,capture,c)
            self.assertEqual(set(d),set(MODES))
            self.assertTrue(torch.equal(z,m.cached_base_seen_logits(d['native'])))
            self.assertTrue(torch.equal(d['train_scaled_dynamic'],.15*d['native']))
            self.assertGreater(float(g[0,0]),float(g[1,0]))
        self.assertEqual(before,tensor_state_digest(m.state_dict()))
        for unit in m.interaction_bank:
            self.assertFalse(unit._forward_hooks);self.assertFalse(unit.out._forward_hooks)
            self.assertFalse(unit.gate._forward_hooks)

    def test_all_common_budget_modes_obey_one_percent_cap(self):
        m,x,c=self.make()
        with ReadoutCapture(m) as capture,torch.no_grad():
            m.seen_logits_with_views(x);d,g,ratios=candidate_deltas(m,capture,c)
            for name,value in ratios.items():
                if name.startswith('cap1'):self.assertLessEqual(float(value.max()),.010001)
            self.assertTrue(torch.equal(d['cap1_constant'],.4*d['cap1_open']))
            self.assertTrue(torch.equal(d['cap1_dynamic'],g[...,None]*d['cap1_open']))

    def test_face_memory_intervention_keeps_recipient_gate(self):
        m,x,c=self.make()
        with ReadoutCapture(m) as capture,torch.no_grad():
            m.seen_logits_with_views(x);d,g,_=candidate_deltas(m,capture,c)
            no_face=capture.tensors([0])
            bound=bounded_delta(no_face['raw'],m._interaction_cache['base'],.01)
            self.assertTrue(torch.equal(d['cap1_dynamic_no_face_fixed_gate'],g[...,None]*bound))
            self.assertGreater(float((g-no_face['gate_logit'].sigmoid()[...,0]).abs().max()),0)
            invalid=~x['face_reliable']
            self.assertTrue(torch.equal(d['cap1_dynamic'][invalid],d['cap1_dynamic_no_face_fixed_gate'][invalid]))

    def test_permutation_preserves_gate_distribution_and_quality_groups(self):
        m,x,c=self.make()
        with ReadoutCapture(m) as capture,torch.no_grad():
            m.seen_logits_with_views(x);d,g,_=candidate_deltas(m,capture,c)
            perm=m._interaction_cache['diagnostic_permuted_gate']
            for reliable in (True,False):
                mask=x['face_reliable']==reliable
                self.assertTrue(torch.equal(g[mask].sort(0).values,perm[mask].sort(0).values))
            # This fixture's gate depends only on the preserved quality flag.
            self.assertTrue(torch.equal(d['cap1_dynamic'],d['cap1_permuted_gate']))

    def test_exception_removes_hooks_and_grad_enabled_use_is_rejected(self):
        m,x,c=self.make()
        with ReadoutCapture(m) as capture,torch.no_grad():m.seen_logits_with_views(x)
        with self.assertRaises(RuntimeError):candidate_deltas(m,capture,c)
        try:
            with ReadoutCapture(m):raise ValueError('abort')
        except ValueError:pass
        self.assertTrue(all(not u._forward_hooks for u in m.interaction_bank))


if __name__=='__main__':unittest.main()
