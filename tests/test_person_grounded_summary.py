import copy
import unittest

import torch

from multi_lane.track_a.person_grounded_summary import (
    GROUNDED_ARMS, MEMORIES, GroundedReadout, PersonGroundedModel, patch_coverage)
from multi_lane.track_a.private_summarize_attention import PrivateSummaryReadout
from multi_lane.track_a.protected_joint_attention import ProtectedJointModel, PrivateInteractionTraining
from multi_lane.track_a.protected_joint_experiment import configure_current, optimizer_groups
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest, train_task
from multi_lane.track_a.task_recovery import rng_state, restore_rng
from test_track_a_reproduction import FakeVisual


class PersonGroundedTests(unittest.TestCase):
    def make(self, arm):
        torch.manual_seed(293)
        cls = ProtectedJointModel if arm == 'C0' else PersonGroundedModel
        m = cls(FakeVisual(), TASK_SIZES, num_selectors=10, num_prompts=2, num_prompt_layers=1,
            view_fusion='fixed_three_view', adapter_mode='image_token', adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,), adapter_residual_scale=.03)
        before = torch.get_rng_state(); m.initialize_interaction(arm, layer=0, dim=4, heads=2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        m.activate_task(0); configure_current(m, 0)
        x = {v: torch.randn(4, 3, 4, 4) for v in ('full', 'person', 'face')}
        x.update(face_reliable=torch.tensor([True, False, True, False]),
            spatial_visibility=torch.tensor([1., 1., .4, 0.]),
            spatial_bbox=torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.], [0., 0., 1., 1.], [2., 2., 3., 3.]]))
        y = torch.tensor([[1., 0., 0., 1., 0.], [0., 1., 0., 0., 1.],
                          [0., 0., 1., 0., 1.], [1., 0., 1., 0., 0.]])
        return m, (x, y)

    def train(self, m, batch, **kwargs):
        return train_task(m, [batch], None, torch.device('cpu'), 0, 3, .001, 0., 1., False,
            adapter_learning_rate=.0004, loss_routing='adapter_asl', view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups,
            private_interaction=PrivateInteractionTraining(m, epochs=3, amp=False) if m.interaction_bank else None,
            **kwargs)

    def source(self):
        tokens = {v: torch.randn(4, 5, 8, requires_grad=True) for v in ('full', 'person', 'face')}
        anchors = {v: torch.randn(4, 10, 8, requires_grad=True) for v in tokens}
        p = torch.randn(4, 8, requires_grad=True)
        boxes = torch.tensor([[0., 0., .5, 1.]]*4); vis = torch.ones(4)
        return tokens, anchors, p, boxes, vis

    def test_geometry_overlap_and_bad_or_invisible_boxes(self):
        b = torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.], [-2., -2., -1., -1.],
                          [float('nan'), 0., 1., 1.], [0., 0., 1., 1.]])
        rho, valid, eligible = patch_coverage(b, torch.tensor([1., .5, 1., 1., 0.]), 4)
        self.assertTrue(torch.equal(rho[0], torch.tensor([1., 0., 1., 0.])))
        self.assertTrue(torch.equal(rho[1], torch.tensor([0., 1., 0., 1.])))
        self.assertEqual(valid.tolist(), [True, True, False, False, False])
        self.assertEqual(eligible.tolist(), [True, False, False, False, False])
        self.assertTrue(torch.isfinite(rho).all()); self.assertEqual(float(rho[2:].sum()), 0.)

    def test_parameter_counts_and_common_initialization(self):
        common = None
        for arm in GROUNDED_ARMS:
            torch.manual_seed(930100); m = GroundedReadout(768, 512, 10, arm)
            expected = 221728 if arm.endswith('ATTN') else 221789
            self.assertEqual(sum(p.numel() for p in m.parameters()), expected)
            values = {k: v for k, v in m.state_dict().items() if not k.startswith(('key.', 'pool.'))}
            if common is None: common = values
            else: self.assertTrue(all(torch.equal(v, common[k]) for k, v in values.items()))

    def test_identity_initialization_and_exact_original_training(self):
        m, batch = self.make('C0'); reference = self.train(m, batch)
        base = lambda model: {k: v for k, v in model.state_dict().items() if not k.startswith('interaction_bank.')}
        expected, rng = tensor_state_digest(base(m)), torch.get_rng_state()
        for arm in GROUNDED_ARMS:
            m, batch = self.make(arm)
            m.eval()
            with torch.no_grad():
                z, _ = m.seen_logits_with_views(batch[0]); self.assertTrue(torch.equal(z, m.cached_base_seen_logits()))
            rows = self.train(m, batch)
            self.assertEqual(expected, tensor_state_digest(base(m)), arm)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), arm)
            self.assertEqual([r['current_loss'] for r in rows], [r['current_loss'] for r in reference])
            self.assertGreater(rows[-1]['interaction_gradient_norm'], 0)
            self.assertGreater(rows[-1]['interaction_query_gradient_norm'], 0)
            self.assertLessEqual(rows[-1]['interaction_delta_to_base_rms_max'], .010001)
            self.assertEqual(rows[0]['interaction_localization_weight'], .002 if '_GROUND_' in arm else 0.)
            self.assertEqual(rows[-1]['interaction_summarize_fp32'], 1.)

    def test_location_objective_updates_only_subject_parameters(self):
        m = GroundedReadout(8, 8, 10, 'PERSON_GROUND_ATTN', 4, 2)
        tokens, anchors, p, boxes, vis = self.source()
        m.summarize(tokens, anchors, p, boxes, vis); m.location_loss.backward()
        self.assertGreater(float(m.subject_offset.grad.norm()), 0.)
        self.assertGreater(float(m.condition_output.grad.norm()), 0.)
        self.assertTrue(all(x.grad is None for x in (*tokens.values(), *anchors.values(), p)))
        for name, parameter in m.named_parameters():
            if name.startswith(('query.', 'value.', 'key.', 'out.', 'pool.', 'view_types')):
                self.assertIsNone(parameter.grad, name)

    def test_global_evidence_remains_and_subject_excludes_frozen_cls(self):
        m = GroundedReadout(8, 8, 10, 'PERSON_CONTENT_ATTN', 4, 2)
        tokens, anchors, p, b, v = self.source()
        a = m.summarize(tokens, anchors, p, b, v)
        changed = {k: x.detach().clone() for k, x in tokens.items()}; changed['full'][:, 0] += 1000
        c = m.summarize(changed, anchors, p, b, v)
        self.assertTrue(torch.equal(a['subject'], c['subject']))
        self.assertFalse(torch.equal(a['global'], c['global']))
        self.assertEqual([a[k].shape[1] for k in MEMORIES], [8, 2, 10, 10])

    def test_person_condition_changes_subject_only_and_constant_control_does_not(self):
        tokens, anchors, p, b, v = self.source(); donor = torch.tensor([1, 0, 3, 2])
        for arm in ('PERSON_CONST_ATTN', 'PERSON_CONTENT_ATTN'):
            m = GroundedReadout(8, 8, 10, arm, 4, 2)
            with torch.no_grad(): m.condition_output.normal_()
            a = m.summarize(tokens, anchors, p, b, v)
            c = m.summarize(tokens, anchors, p[donor], b, v)
            for name in ('global', 'person', 'face'): self.assertTrue(torch.equal(a[name], c[name]))
            self.assertEqual(torch.equal(a['subject'], c['subject']), '_CONST_' in arm)

    def test_private_backward_has_no_original_or_head_gradients(self):
        for arm in GROUNDED_ARMS:
            m, (x, y) = self.make(arm); m.train(); m.current_all_logits_with_views(x)
            private = PrivateInteractionTraining(m, amp=False); private.begin_epoch(); private.backward(y, range(5))
            self.assertTrue(all(p.grad is None for n, p in m.named_parameters() if not n.startswith('interaction_bank.0.')))

    def test_interventions_hold_late_query_and_off_and_bad_face_exact(self):
        for arm in GROUNDED_ARMS:
            m, (x, y) = self.make(arm); self.train(m, (x, y)); m.eval(); queries = []
            with torch.no_grad():
                z, views = m.seen_logits_with_views(x); off = m.cached_base_seen_logits()
                hook = m.interaction_bank[0].query.register_forward_pre_hook(lambda _, a: queries.append(a[0].clone()))
                native = m.interaction_delta()
                m.interaction_delta(early_donor=torch.tensor([1, 0, 3, 2]))
                no_face = m.interaction_delta(no_face=True); m.interaction_delta(no_subject=True)
                m.interaction_delta(no_global=True); m.interaction_delta(no_person=True); hook.remove()
                self.assertTrue(all(torch.equal(queries[0], q) for q in queries[1:]))
                self.assertTrue(torch.equal(native[~x['face_reliable']], no_face[~x['face_reliable']]))
                changed = copy.deepcopy(x); changed['face'][~x['face_reliable']] *= 11
                a, _ = m.seen_logits_with_views(changed)
                self.assertTrue(torch.equal(z[~x['face_reliable']], a[~x['face_reliable']]))
                m.interaction_enabled = False; b, vv = m.seen_logits_with_views(x)
                self.assertTrue(torch.equal(off, b)); self.assertTrue(all(torch.equal(views[v], vv[v]) for v in views))

    def test_epoch_recovery_both_optimizers_and_warmup(self):
        for arm in ('PERSON_GROUND_POOL', 'PERSON_GROUND_ATTN'):
            m, batch = self.make(arm); saved = {}
            def capture(epoch, rows, state): saved[epoch] = (copy.deepcopy(m.state_dict()), copy.deepcopy(state), rng_state())
            rows = self.train(m, batch, epoch_callback=capture); expected = tensor_state_digest(m.state_dict())
            n, batch = self.make(arm); weights, state, rng = saved[1]; n.load_state_dict(weights); restore_rng(rng)
            actual = self.train(n, batch, resume_training_state=state)
            self.assertEqual(expected, tensor_state_digest(n.state_dict()))
            for key in ('current_loss', 'interaction_loss', 'interaction_localization_weight'):
                self.assertEqual([r[key] for r in rows], [r[key] for r in actual])

    def test_no_eligible_people_is_finite_zero_and_keeps_global(self):
        m = GroundedReadout(8, 8, 10, 'PERSON_GROUND_ATTN', 4, 2)
        tokens, anchors, p, b, v = self.source(); v.zero_()
        s = m.summarize(tokens, anchors, p, b, v)
        self.assertEqual(float(m.location_loss), 0.); self.assertFalse(m.subject_valid.any())
        self.assertGreater(float(s['global'].norm()), 0.)
        m.location_loss.backward(); self.assertTrue(torch.isfinite(m.subject_offset.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP contract requires GPU')
    def test_cuda_amp_preserves_fp32_for_new_and_repaired_summarize(self):
        tokens, anchors, p, b, v = self.source()
        tokens = {k: x.detach().cuda().half() for k, x in tokens.items()}
        anchors = {k: x.detach().cuda().half() for k, x in anchors.items()}
        m = GroundedReadout(8, 8, 10, 'PERSON_GROUND_ATTN', 4, 2).cuda()
        old = PrivateSummaryReadout(8, 8, 10, 'SUMMARY_LEARN_ATTN', 4, 2).cuda()
        with torch.cuda.amp.autocast(enabled=True):
            s = m.summarize(tokens, anchors, p.cuda(), b.cuda(), v.cuda())
            repaired = old.summarize(tokens, anchors)
            z = m(torch.randn(4, 8, device='cuda'), p.cuda(), s, torch.ones(4, device='cuda', dtype=torch.bool))
        self.assertTrue(all(x.dtype == torch.float32 for x in (*s.values(), *repaired.values(), z)))
        self.assertEqual(m.location_loss.dtype, torch.float32)


if __name__ == '__main__': unittest.main()
