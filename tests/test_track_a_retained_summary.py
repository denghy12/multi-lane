import unittest
from unittest.mock import patch

import torch

import test_track_a_joint_summary as fixtures
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.evaluate_moe_task_ffn import evaluate_split
from pathlib import Path


class RetainedSummaryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.JointSummaryTest()
        self.model = self.fixture.model('retained')
        self.model.activate_task(0)

    def test_single_cls_reads_three_groups_but_restores_original_selectors(self):
        model = self.model
        model.eval()
        rows = []
        original = model._task_attention_ffn
        def capture(block, lanes, ids, layer, cls, selectors, selected, *args, **kwargs):
            selected.retain_grad()
            rows.append((lanes.shape, selected, kwargs['summary_valid'].clone()))
            return original(block, lanes, ids, layer, cls, selectors, selected, *args, **kwargs)
        images = self.fixture.images()
        with patch.object(model, '_task_attention_ffn', side_effect=capture):
            model.current_logits(images).square().sum().backward()
        self.assertEqual(model.joint_summary_bank.parameter_count_per_task(), 0)
        self.assertEqual(len(list(model.joint_summary_bank.parameters())), 0)
        self.assertEqual(len(rows), 2)
        for lanes, selected, mask in rows:
            self.assertEqual(tuple(lanes), (1, 3, 3, 8))
            self.assertEqual(tuple(selected.shape), (1, 3, 6, 8))
            self.assertEqual(mask.tolist(), [[True]*7, [True]*5+[False]*2, [True]*7])
            self.assertTrue(torch.equal(selected.grad[0, 1, 4:], torch.zeros_like(selected.grad[0, 1, 4:])))
            self.assertGreater(selected.grad[0, 0, :2].abs().sum(), 0)
            self.assertGreater(selected.grad[0, 0, 2:4].abs().sum(), 0)
            self.assertGreater(selected.grad[0, 0, 4:].abs().sum(), 0)
        self.assertGreater(model.selectors.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_invalid_keys_are_removed_before_softmax_including_prefix_layers(self):
        model = self.model
        model.eval()
        for layer, block in enumerate(model.visual_encoder.transformer.resblocks):
            tokens = torch.randn(1, 3, 7, model.width)
            valid = torch.tensor([[True]*5+[False]*2]*3)
            actual = model._prompt_attention(block, tokens, [0], layer, key_valid=valid)
            expected = model._prompt_attention(block, tokens[:, :, :5], [0], layer)
            self.assertTrue(torch.allclose(actual[:, :, :1], expected[:, :, :1], atol=1e-6))
            tokens[:, :, 5:] *= 1000
            changed = model._prompt_attention(block, tokens, [0], layer, key_valid=valid)
            self.assertTrue(torch.equal(actual[:, :, :1], changed[:, :, :1]))
        images = self.fixture.images()
        images['face_reliable'].fill_(False)
        before = model.seen_logits(images)
        images['face'] = torch.randn_like(images['face']) * 100
        self.assertTrue(torch.equal(before, model.seen_logits(images)))

    def test_full_only_equivalence_old_task_protection_and_compact_restore(self):
        model = self.model
        model.eval()
        images = self.fixture.images()
        images['joint_view_valid'] = torch.tensor([[True, False, False]]*3)
        with torch.no_grad():
            before = model.seen_logits(images)
            full = model._encode_single_lanes(images['full'], False)
            self.assertTrue(torch.allclose(model.encode_lanes(images, False), full, atol=1e-6))
        model.activate_task(1)
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            model.current_logits(images).square().sum().backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            after = model.seen_logits(images)
            self.assertTrue(torch.allclose(before, after[:, :2], atol=1e-6))
        restored = self.fixture.model('retained')
        restore_compact_model_state(restored, dict(schema_version=1, task_id=1,
            source_git={'commit':'unit'}, model=compact_model_state_dict(model)), 1, {'commit':'unit'})
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(after, restored.seen_logits(images)))

    def test_residual_ablation_cannot_run_on_test(self):
        kwargs = dict(source_run=Path('absent'), output=Path('absent'), split='test',
                      data_root=Path('.'), clip_checkpoint=Path('.'),
                      face_manifest_root=Path('.'), batch_size=64)
        for flags in ({'checkpoint_tasks':[0,2,7]}, {'disable_joint_residual':True}):
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                evaluate_split(**kwargs, **flags)


if __name__ == '__main__':
    unittest.main()
