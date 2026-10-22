import copy
import unittest
import torch

from multi_lane.track_a.spatial_summary import TaskSpatialSummary
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
import test_track_a_spatial_summary as spatial_tests


class RegionalSummaryTest(unittest.TestCase):
    def fixture(self):
        helper = spatial_tests.SpatialSummaryTest()
        model = helper.model('regional')
        model.activate_task(0)
        return model, helper.images()

    def test_disjoint_regions_have_explicit_mass_and_bounded_quota(self):
        bank = TaskSpatialSummary(2, 2, 'regional')
        logits = torch.zeros(1, 1, 2, 5)
        overlap = torch.tensor([[0., 1., 0., 0., 0.]])
        a = bank.regional_attention(logits, overlap, torch.ones(1), [0], 0)
        self.assertTrue(torch.allclose(a.sum(-1), torch.ones(1, 1, 2)))
        self.assertTrue(torch.allclose(a[..., 1], torch.full((1, 1, 2), .35)))
        self.assertTrue(torch.allclose(a[..., 0], torch.full((1, 1, 2), .1)))
        self.assertTrue((a >= .5 * logits.softmax(-1)).all())
        for value, expected in [(-100., .225), (100., .475)]:
            with torch.no_grad(): bank.tasks[0].fill_(value)
            a = bank.regional_attention(logits, overlap, torch.ones(1), [0], 0)
            self.assertTrue(torch.allclose(a[..., 1], torch.full((1, 1, 2), expected)))

    def test_fractional_masks_extreme_logits_and_empty_regions_never_nan(self):
        bank = TaskSpatialSummary(1, 2, 'regional')
        logits = torch.tensor([[[[1000., -1000., 5., -5., 0.], [0., 1., 2., 3., 4.]]]])
        overlap = torch.tensor([[0., .2, .8, 0., 0.]])
        a = bank.regional_attention(logits, overlap, torch.ones(1), [0], 0)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.allclose(a.sum(-1), torch.ones(1, 1, 2), atol=1e-6))
        for mask, visibility in [(torch.zeros_like(overlap), 1.),
                                 (torch.tensor([[0., 1., 1., 1., 1.]]), 1.),
                                 (overlap, 0.), (torch.tensor([[0., .1, 0., 0., 0.]]), 1.)]:
            a = bank.regional_attention(logits, mask, torch.tensor([visibility]), [0], 0)
            self.assertTrue(torch.equal(a, logits.softmax(-1)))

    def test_runtime_off_is_exact_base_and_person_face_unchanged(self):
        helper = spatial_tests.SpatialSummaryTest()
        base = helper.model('disabled'); base.activate_task(0)
        model, images = self.fixture()
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        with torch.no_grad():
            expected, base_views = base.seen_logits_with_views(images)
            normal, views = model.seen_logits_with_views(images)
            self.assertFalse(torch.equal(views['full'][:2], base_views['full'][:2]))
            for name in ('person', 'face'):
                self.assertTrue(torch.equal(views[name], base_views[name]))
            self.assertTrue(torch.equal(normal[2], expected[2]))
            swapped, alt = model.seen_logits_with_views(dict(images, spatial_bbox=images['spatial_bbox'].roll(1, 0)))
            self.assertFalse(torch.equal(normal[:2], swapped[:2]))
            for name in ('person', 'face'):
                self.assertTrue(torch.equal(views[name], alt[name]))
            model.spatial_summary_runtime_enabled = False
            self.assertTrue(torch.equal(expected, model.seen_logits(images)))

    def test_gradients_only_current_quota_and_original_frozen_backbone(self):
        model, images = self.fixture()
        before = tensor_state_digest(model.visual_encoder.state_dict())
        groups = build_optimizer_groups(model, 0, view_fusion_learning_rate=.0004)[2]
        self.assertEqual(groups[-1]['lr'], .0004)
        self.assertEqual(model.spatial_summary_bank.parameter_count_per_task(), 2)
        optimizer = torch.optim.Adam(groups, lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward()
        grad = model.spatial_summary_bank.tasks[0].grad
        self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(grad.abs().sum(), 0)
        self.assertIsNone(model.spatial_summary_bank.tasks[1].grad)
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
        optimizer.step()
        self.assertEqual(before, tensor_state_digest(model.visual_encoder.state_dict()))

    def test_old_task_and_compact_roundtrip(self):
        model, images = self.fixture()
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): old = model.seen_logits(images)
        quota = model.spatial_summary_bank.tasks[0].detach().clone()
        model.activate_task(1)
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward(); optimizer.step()
        self.assertTrue(torch.equal(quota, model.spatial_summary_bank.tasks[0]))
        self.assertIsNone(model.spatial_summary_bank.tasks[0].grad)
        model.eval()
        with torch.no_grad(): expected = model.seen_logits(images)
        self.assertTrue(torch.allclose(expected[:, :2], old, atol=1e-6))
        restored, _ = self.fixture()
        restore_compact_model_state(restored, dict(schema_version=1, task_id=1,
            source_git={'commit': 'unit'}, model=compact_model_state_dict(model)), 1, {'commit':'unit'})
        with torch.no_grad(): self.assertTrue(torch.equal(expected, restored.seen_logits(images)))

    def test_invalid_geometry_exact_fallback_and_same_rng(self):
        helper = spatial_tests.SpatialSummaryTest()
        base = helper.model('disabled'); state = torch.get_rng_state()
        model = helper.model('regional')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        base.activate_task(0); model.activate_task(0)
        images = helper.images()
        images['spatial_bbox'][:] = float('nan')
        with torch.no_grad(): self.assertTrue(torch.equal(base.seen_logits(images), model.seen_logits(images)))


if __name__ == '__main__':
    unittest.main()
