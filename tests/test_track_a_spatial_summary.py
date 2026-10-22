import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from PIL import Image

from multi_lane.track_a.spatial_summary import TaskSpatialSummary
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.paired_transforms import PairedFullPersonTransform
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
import test_track_a_reproduction as fixtures


class SpatialSummaryTest(unittest.TestCase):
    def model(self, mode="target"):
        torch.manual_seed(71)
        visual = fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual, (2, 1), num_selectors=2, num_prompts=2,
                             num_prompt_layers=1, view_fusion="fixed_three_view", spatial_summary_mode=mode)

    def images(self):
        return {**{name: torch.randn(3, 3, 4, 4) for name in ("full", "person", "face")},
                "face_reliable": torch.tensor([True, False, True]),
                "spatial_bbox": torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.], [0., 0., 1., 1.]]),
                "spatial_visibility": torch.tensor([1., 1., 0.])}

    def test_zero_parameters_preserve_base_rng_and_views(self):
        base = self.model("disabled"); rng = torch.get_rng_state(); base.activate_task(0)
        images = self.images()
        with torch.no_grad(): expected, views = base.seen_logits_with_views(images)
        for mode in ("absolute", "target"):
            model = self.model(mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            model.activate_task(0)
            self.assertEqual(model.spatial_summary_bank.parameter_count_per_task(), 12)
            for name, value in base.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            with torch.no_grad(): actual, new = model.seen_logits_with_views(images)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(torch.equal(views[name], new[name]) for name in views))

    def test_optimizer_covers_only_active_spatial_and_gradients(self):
        for mode in ("absolute", "target"):
            model = self.model(mode); model.activate_task(0)
            groups = build_optimizer_groups(model, 0, view_fusion_learning_rate=.0004)[2]
            self.assertEqual({id(p) for g in groups for p in g['params']},
                             {id(dict(model.named_parameters())[n]) for n in model.optimizer_parameter_names()})
            self.assertEqual(groups[-1]['lr'], .0004)
            before = tensor_state_digest(model.visual_encoder.state_dict())
            optimizer = torch.optim.Adam(groups, lr=.01)
            for _ in range(2):
                optimizer.zero_grad(); model.current_logits(self.images()).square().sum().backward()
                g = model.spatial_summary_bank.tasks[0].grad
                self.assertTrue(torch.isfinite(g).all()); self.assertGreater(g.abs().sum(), 0)
                self.assertIsNone(model.spatial_summary_bank.tasks[1].grad)
                optimizer.step()
            self.assertEqual(before, tensor_state_digest(model.visual_encoder.state_dict()))
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_target_swap_and_runtime_off_only_change_full(self):
        model = self.model(); model.activate_task(0); images = self.images()
        with torch.no_grad(): model.spatial_summary_bank.tasks[0].normal_(0, .8)
        alt = dict(images, spatial_bbox=images['spatial_bbox'].roll(1, 0))
        with torch.no_grad():
            before, views = model.seen_logits_with_views(images)
            after, new = model.seen_logits_with_views(alt)
            self.assertFalse(torch.equal(views['full'][:2], new['full'][:2]))
            self.assertTrue(torch.equal(before[2], after[2]))
            for view in ('person', 'face'): self.assertTrue(torch.equal(views[view], new[view]))
            model.spatial_summary_runtime_enabled = False
            off, off_views = model.seen_logits_with_views(images)
            self.assertFalse(torch.equal(before, off))
            for view in ('person', 'face'): self.assertTrue(torch.equal(views[view], off_views[view]))
            model.spatial_summary_runtime_enabled = True
            self.assertTrue(torch.equal(before, model.seen_logits(images)))

    def test_absolute_control_has_same_visibility_but_no_target_location(self):
        bank = TaskSpatialSummary(1, 2, 'absolute')
        with torch.no_grad(): bank.tasks[0].normal_()
        images = self.images(); b = images['spatial_bbox']; v = images['spatial_visibility']
        a = bank.bias(b, v, 5, [0])[0]
        c = bank.bias(b.roll(1, 0), v, 5, [0])[0]
        self.assertTrue(torch.equal(a, c)); self.assertTrue(torch.equal(a[:, 2], torch.zeros_like(a[:, 2])))
        self.assertLessEqual(a.abs().max(), 1)
        self.assertTrue(torch.equal(a[..., 0], torch.zeros_like(a[..., 0])))

    def test_malformed_geometry_falls_back_and_never_nan(self):
        bank = TaskSpatialSummary(1, 2, 'target')
        with torch.no_grad(): bank.tasks[0].fill_(1)
        b = torch.tensor([[float('nan'), 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1.]])
        value = bank.bias(b, torch.tensor([1., 1., float('inf')]), 5, [0])[0]
        self.assertTrue(torch.equal(value, torch.zeros_like(value)))
        with self.assertRaises(ValueError): bank.bias(torch.zeros(3, 6), torch.ones(3), 5, [0])
        with self.assertRaises(ValueError): bank.bias(torch.zeros(3, 4), torch.ones(3), 6, [0])

    def test_untruncated_geometry_and_flip(self):
        box = (-10, 15, 90, 75); crop = (10, 20, 100, 100)
        p = PairedFullPersonTransform.untruncated_geometry(box, crop, False)
        f = PairedFullPersonTransform.untruncated_geometry(box, crop, True)
        self.assertTrue(torch.allclose(p, torch.tensor([-.3, .05, .7, .65])))
        self.assertTrue(torch.allclose(f, torch.tensor([.3, .05, 1.3, .65])))
        self.assertTrue(torch.equal(PairedFullPersonTransform.untruncated_geometry(None, crop, False), torch.zeros(4)))

    def test_transform_metadata_preserves_pixels_and_rng(self):
        image = Image.new('RGB', (300, 200), (21, 47, 100))
        for train in (False, True):
            options = dict(train=train, margin=.15, jitter_strength=0, jitter_probability=0)
            torch.manual_seed(19); base = PairedFullPersonTransform(**options)(image, [3, 10, 120, 180]); rng = torch.get_rng_state()
            torch.manual_seed(19); extra = PairedFullPersonTransform(**options, emit_spatial_geometry=True)(image, [3, 10, 120, 180])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key in base: self.assertTrue(torch.equal(base[key], extra[key]), key)
            self.assertEqual(extra['spatial_bbox'].shape, (4,))
            self.assertEqual(extra['spatial_visibility'], extra['bbox'][4] * extra['bbox'][5])

    def test_every_full_layer_uses_same_parameter_bank_and_original_lengths(self):
        model = self.model(); model.activate_task(0); calls = []
        original = model._task_attention_ffn
        def record(*args, **kwargs):
            calls.append((args[3], args[6].shape[-2], args[7]))
            return original(*args, **kwargs)
        with patch.object(model, '_task_attention_ffn', side_effect=record): model.seen_logits(self.images())
        self.assertEqual(calls, [(layer, 2, name) for name in ('full', 'person', 'face') for layer in (0, 1)])
        self.assertEqual(set(model.spatial_summary_bank.statistics()), {'task0/layer0', 'task0/layer1'})

    def test_old_task_protection_and_compact_roundtrip(self):
        model = self.model(); model.activate_task(0); images = self.images()
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): old = model.seen_logits(images)
        old_weight = model.spatial_summary_bank.tasks[0].detach().clone()
        model.activate_task(1)
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward(); optimizer.step()
        self.assertTrue(torch.equal(old_weight, model.spatial_summary_bank.tasks[0]))
        with torch.no_grad(): new = model.seen_logits(images)
        self.assertTrue(torch.allclose(old, new[:, :2], atol=1e-6))
        restored = self.model()
        restore_compact_model_state(restored, dict(schema_version=1, task_id=1, source_git={'commit': 'unit'},
                                  model=compact_model_state_dict(model)), 1, {'commit': 'unit'})
        with torch.no_grad(): self.assertTrue(torch.equal(new, restored.seen_logits(images)))

    def test_missing_geometry_and_incompatible_modes_rejected(self):
        model = self.model(); model.activate_task(0); images = self.images(); del images['spatial_bbox']
        with self.assertRaises(ValueError): model.seen_logits(images)
        with self.assertRaises(ValueError): TaskSpatialSummary(1, 2, 'bad')
        with self.assertRaises(ValueError):
            MultiLaneModel(fixtures.FakeVisual(), (2, 1), spatial_summary_mode='target')

    def test_attribution_dataset_changes_only_box_and_keeps_recipient_visibility(self):
        from multi_lane.track_a.evaluate_spatial_attribution import SpatialAttributionDataset
        from multi_lane.track_a.cross_view_query_pairs import make_pairs
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.png'
            Image.new('RGB', (200, 200), (21, 40, 75)).save(path)
            class Source:
                file_paths = [str(path), str(path)]
                body_bboxes = [[30,30,80,180], [120,20,170,170]]
                sample_ids = ['val:scene#person=0', 'val:scene#person=1']
                def __len__(self): return 2
                def __getitem__(self, index):
                    with Image.open(path) as image:
                        images = PairedFullPersonTransform(train=False, emit_spatial_geometry=True)(image, self.body_bboxes[index])
                    images.update(face=torch.zeros_like(images['full']), face_reliable=torch.tensor(True))
                    return images, torch.tensor([1., 0.])
            source = Source()
            plan = make_pairs(source.sample_ids, source.body_bboxes, [(200,200)]*2)
            self.assertEqual(plan['eligible_people'], 2)
            normal = SpatialAttributionDataset(source, plan, [0])
            swapped = SpatialAttributionDataset(source, plan, [0], True)
            a, y, sid, index = normal[0]; b, alt_y, alt_sid, alt_index = swapped[0]
            self.assertFalse(torch.equal(a['spatial_bbox'], b['spatial_bbox']))
            for key in a:
                if key != 'spatial_bbox': self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertTrue(torch.equal(y, alt_y)); self.assertEqual((sid,index), (alt_sid,alt_index))

    def test_complete_recovery_loader_accepts_version_and_rng_metadata(self):
        import numpy as np
        from multi_lane.track_a.verify_spatial_summary import load_boundary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'boundary.pth'
            state = dict(torch_version=torch.__version__, numpy_rng=np.random.get_state(),
                         model={'weight': torch.ones(2)})
            torch.save(state, path)
            actual = load_boundary(path)
            self.assertEqual(actual['torch_version'], state['torch_version'])
            self.assertTrue(np.array_equal(actual['numpy_rng'][1], state['numpy_rng'][1]))
            self.assertTrue(torch.equal(actual['model']['weight'], state['model']['weight']))


if __name__ == '__main__': unittest.main()
