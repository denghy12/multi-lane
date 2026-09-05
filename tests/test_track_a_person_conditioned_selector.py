from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from test_track_a_reproduction import FakeVisual
from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.paired_transforms import PairedFullPersonTransform
from multi_lane.track_a.selector_conditioning import TaskSelectorConditioner
from multi_lane.track_a.runner import (
    LabelView, backward_routed_training_losses, build_optimizer_groups,
    build_transforms, compact_model_state_dict, compute_asymmetric_training_loss,
    compute_training_loss, evaluate, train_task,
)


def tiny_model(mode="person", seed=17):
    torch.manual_seed(seed)
    return MultiLaneModel(
        FakeVisual(), (5, 3), num_selectors=2, num_prompts=2,
        num_prompt_layers=1, adapter_mode="image_token",
        adapter_layer_indices=(0,), adapter_bottleneck_dim=3,
        selector_conditioning=mode, selector_condition_layers=(0,),
        selector_condition_hidden_dim=4,
    )


def paired_batch():
    return {"full": torch.randn(3, 3, 4, 4),
            "person": torch.randn(3, 3, 4, 4),
            "bbox": torch.tensor([[.1, .2, .8, .9, 1, 1]]).repeat(3, 1),
            "person_patch_mask": torch.ones(3, 4),
            "condition_valid": torch.ones(3)}


class PersonConditionedSelectorTest(unittest.TestCase):
    def test_all_modes_preserve_initial_logits_base_parameters_and_rng(self):
        baseline = tiny_model("disabled")
        state = torch.get_rng_state().clone()
        baseline.activate_task(0)
        inputs = paired_batch()
        with torch.no_grad():
            expected = baseline.current_all_logits(inputs["full"])
        for mode in ("bbox", "person", "bbox_person", "person_patches"):
            with self.subTest(mode=mode):
                model = tiny_model(mode)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                for name, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                model.activate_task(0)
                with torch.no_grad():
                    actual = model.current_all_logits(inputs)
                self.assertTrue(torch.equal(expected, actual))

    def test_person_patch_attention_is_selector_specific_and_masks_padding(self):
        conditioner = TaskSelectorConditioner(
            1, 8, "person_patches", (0,), hidden_dim=4, residual_scale=1
        )
        conditioner.restore_task(0)
        mlp = conditioner.task_modules[0]["0"]
        with torch.no_grad():
            mlp[0].weight.zero_()
            mlp[0].bias.zero_()
            mlp[2].weight.zero_()
            mlp[2].bias.zero_()
            mlp[0].weight[:, :4] = torch.eye(4)
            mlp[2].weight[:4] = torch.eye(4)
        selectors = torch.zeros(1, 1, 2, 8)
        selectors[0, 0, 0, 0] = 8
        selectors[0, 0, 1, 1] = 8
        patches = torch.zeros(1, 3, 8)
        patches[0, 0, 0] = 8
        patches[0, 1, 1] = 8
        patches[0, 2] = 1_000
        mask = torch.tensor([[True, True, False]])
        expected = conditioner.patch_query_delta(
            0, (0,), selectors, patches, mask, torch.ones(1)
        )
        patches[0, 2] = -1_000
        actual = conditioner.patch_query_delta(
            0, (0,), selectors, patches, mask, torch.ones(1)
        )
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(actual[0, 0, 0], actual[0, 0, 1]))

    def test_condition_queries_do_not_replace_persistent_selectors(self):
        model = tiny_model()
        model.activate_task(0)
        selectors = model.selectors.detach().clone()
        with torch.no_grad():
            model.selector_conditioner.task_modules[0]["0"][2].weight.normal_(0, .2)
        inputs = paired_batch()
        seen_queries = []
        original = model._prompt_attention

        def capture(block, summarized, lane_ids, layer_id):
            seen_queries.append(summarized.detach().clone())
            return original(block, summarized, lane_ids, layer_id)

        with patch.object(model, "_prompt_attention", side_effect=capture):
            first = model.current_all_logits(inputs)
            inputs["person"] = torch.randn_like(inputs["person"]) * 3
            second = model.current_all_logits(inputs)
        self.assertFalse(torch.equal(seen_queries[0], seen_queries[1]))
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(torch.equal(selectors, model.selectors))

    def test_invalid_target_disables_even_nonzero_condition(self):
        model = tiny_model("bbox_person")
        model.activate_task(0)
        with torch.no_grad():
            model.selector_conditioner.task_modules[0]["0"][2].weight.fill_(.25)
            model.selector_conditioner.task_modules[0]["0"][2].bias.fill_(1)
        inputs = paired_batch()
        inputs["condition_valid"].zero_()
        actual = model.current_all_logits(inputs)
        model.set_selector_conditioning_runtime_enabled(False)
        self.assertTrue(torch.equal(actual, model.current_all_logits(inputs["full"])))

    def test_person_patch_mode_uses_frozen_same_depth_tokens(self):
        model = tiny_model("person_patches")
        model.activate_task(0)
        with torch.no_grad():
            model.selector_conditioner.task_modules[0]["0"][2].weight.normal_(0, .2)
        inputs = paired_batch()
        inputs["person"].requires_grad_(True)
        first = model.current_all_logits(inputs)
        inputs["person"] = torch.randn_like(inputs["person"])
        second = model.current_all_logits(inputs)
        self.assertFalse(torch.equal(first, second))
        self.assertIsNone(inputs["person"].grad)
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_bce_routes_to_conditioner_asl_to_adapter_and_frozen_person(self):
        model = tiny_model("bbox_person")
        model.activate_task(0)
        inputs = paired_batch()
        inputs["person"].requires_grad_(True)
        logits = model.current_all_logits(inputs)
        targets = torch.tensor([[1, 0, 1, 0, 0]]).repeat(3, 1).float()
        bce = compute_training_loss(logits, targets, range(5), 1, "legacy_full_zero")
        asl = compute_asymmetric_training_loss(logits, targets, range(5), 1, "legacy_full_zero")
        base, adapter, groups = build_optimizer_groups(model, 0, 4e-4,
                                                       selector_condition_learning_rate=2e-4)
        condition = list(model.conditioning_optimizer_parameters())
        expected = torch.autograd.grad(bce, condition, retain_graph=True)
        expected_adapter = torch.autograd.grad(asl, adapter, retain_graph=True)
        backward_routed_training_losses(bce, asl, base, adapter,
                                        torch.cuda.amp.GradScaler(enabled=False), False)
        for parameter, gradient in zip(condition, expected):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        for parameter, gradient in zip(adapter, expected_adapter):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        self.assertTrue(any(torch.count_nonzero(p.grad) for p in condition))
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
        self.assertIsNone(inputs["person"].grad)
        self.assertEqual(groups[-2]["lr"], 2e-4)
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.optimizer_parameters()})

    def test_old_task_condition_is_frozen_and_checkpoint_restores_all_seen_lanes(self):
        model = tiny_model()
        model.activate_task(0)
        with torch.no_grad():
            model.selector_conditioner.task_modules[0]["0"][2].weight.normal_()
        old = copy.deepcopy(model.selector_conditioner.task_modules[0].state_dict())
        model.activate_task(1)
        self.assertTrue(all(not p.requires_grad for p in
                            model.selector_conditioner.task_modules[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.conditioning_optimizer_parameters()))
        self.assertEqual(torch.count_nonzero(
            model.selector_conditioner.task_modules[1]["0"][2].weight).item(), 0)
        inputs = paired_batch()
        optimizer = torch.optim.Adam(model.optimizer_parameters(), lr=.001)
        model.current_all_logits(inputs).sum().backward()
        optimizer.step()
        for key, value in old.items():
            self.assertTrue(torch.equal(value, model.selector_conditioner.task_modules[0].state_dict()[key]))
        restored = tiny_model()
        restored.load_state_dict(compact_model_state_dict(model), strict=False)
        restored.restore_task(1)
        self.assertTrue(torch.equal(model.seen_logits(inputs), restored.seen_logits(inputs)))
        self.assertEqual(tuple(restored.seen_logits(inputs).shape), (3, 8))
        names = set(restored.optimizer_parameter_names())
        actual_names = {n for n, p in restored.named_parameters()
                        if id(p) in {id(x) for x in restored.optimizer_parameters()}}
        self.assertEqual(names, actual_names)

    def test_missing_or_misaligned_pairs_fail_closed(self):
        model = tiny_model()
        model.activate_task(0)
        with self.assertRaisesRegex(ValueError, "paired"):
            model.current_all_logits(torch.randn(3, 3, 4, 4))
        batch = paired_batch()
        batch["person"] = batch["person"][:2]
        with self.assertRaisesRegex(ValueError, "shapes"):
            model.current_all_logits(batch)
        with self.assertRaises(ValueError):
            MultiLaneModel(FakeVisual(), (5, 3), num_prompt_layers=1,
                           selector_conditioning="person", selector_condition_layers=(1,))

    def test_paired_batch_runs_training_validation_and_score_dump(self):
        model = tiny_model("bbox_person")
        model.activate_task(0)
        batch = paired_batch()
        labels = torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0]]).float()
        items = [({k: v[i] for k, v in batch.items()}, labels[i]) for i in range(3)]
        loader = DataLoader(items, batch_size=2)
        history = train_task(model, loader, loader, torch.device("cpu"),
                             0, 1, .0125, 0, 1, False,
                             adapter_learning_rate=4e-4, loss_routing="adapter_asl",
                             selector_condition_learning_rate=2e-4)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["selector_condition_learning_rate"], 2e-4)
        self.assertEqual(history[0]["selector_condition_valid_fraction"], 1)
        self.assertEqual(history[0]["next_selector_condition_learning_rate"], 0)
        score_loader = DataLoader([(*item, f"val:a#person={i}")
                                   for i, item in enumerate(items)], batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            row = evaluate(model, score_loader, torch.device("cpu"), 0, .5, False, path)
            self.assertTrue(np.isfinite(row.mAP))
            with np.load(path, allow_pickle=False) as scores:
                self.assertEqual(scores["sample_ids"].tolist(),
                                 [f"val:a#person={i}" for i in range(3)])


class PairedFullPersonInputTest(unittest.TestCase):
    @staticmethod
    def image():
        return Image.fromarray(np.random.default_rng(12).integers(
            0, 256, (193, 307, 3), dtype=np.uint8))

    def test_full_train_pixels_and_rng_match_legacy_across_seeds(self):
        legacy, _ = build_transforms("clip")
        paired = PairedFullPersonTransform(True, jitter_probability=1)
        image = self.image()
        for seed in range(12):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                expected = legacy(image)
                state = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                actual = paired(image, [100, 20, 200, 175])
                self.assertTrue(torch.equal(expected, actual["full"]))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_full_eval_pixels_and_person_letterbox_match_legacy(self):
        _, full_eval = build_transforms("clip")
        _, person_eval = build_transforms("clip", input_mode="person_crop",
                                          person_transform_mode="letterbox")
        cropper = object.__new__(EMOTIC)
        cropper.person_crop_margin = .15
        image = self.image()
        box = [100, 20, 200, 175]
        actual = PairedFullPersonTransform(False)(image, box)
        self.assertTrue(torch.equal(actual["full"], full_eval(image)))
        self.assertTrue(torch.equal(actual["person"], person_eval(cropper._crop_person(image, box))))

    def test_target_aware_train_and_eval_retain_valid_target(self):
        image = self.image()
        box = [100, 20, 200, 175]
        train = PairedFullPersonTransform(
            True, full_crop_mode="target_aware", jitter_probability=0
        )
        for seed in range(20):
            torch.manual_seed(seed)
            pair = train(image, box)
            self.assertEqual(pair["bbox"][4].item(), 1)
            self.assertEqual(pair["condition_valid"].item(), 1)
        evaluate = PairedFullPersonTransform(
            False, full_crop_mode="target_aware", jitter_probability=0
        )
        pair = evaluate(image, box)
        self.assertEqual(pair["bbox"][4].item(), 1)
        self.assertEqual(pair["condition_valid"].item(), 1)

    def test_person_patch_mask_excludes_letterbox_padding(self):
        transform = PairedFullPersonTransform(
            True, normalization="none", margin=0, jitter_probability=0
        )
        image = Image.new("RGB", (100, 200), (128, 128, 128))
        with patch("torchvision.transforms.RandomResizedCrop.get_params",
                   return_value=(0, 0, 200, 100)), patch(
                       "torch.rand", return_value=torch.tensor([1.0])):
            pair = transform(image, [40, 0, 60, 200])
        mask = pair["person_patch_mask"].reshape(14, 14)
        self.assertTrue(mask.any())
        self.assertFalse(mask[:, 0].any())
        self.assertFalse(mask[:, -1].any())
        self.assertTrue(mask[:, 6:8].all())

    def test_geometry_partial_crop_and_flip(self):
        box, crop = (20, 10, 80, 90), (0, 50, 100, 50)
        actual = PairedFullPersonTransform.geometry(box, crop, True)
        torch.testing.assert_close(actual, torch.tensor([.4, .1, 1, .9, .5, 1]))

    def test_outside_and_invalid_boxes_are_masked(self):
        transform = PairedFullPersonTransform(True, jitter_probability=0)
        for box in ([10, 10, 5, 20], [float("nan"), 1, 3, 4], [], [0, 0, 5, 5]):
            with self.subTest(box=box), patch(
                "torchvision.transforms.RandomResizedCrop.get_params",
                return_value=(50, 50, 100, 100),
            ):
                pair = transform(self.image(), box)
                self.assertEqual(pair["condition_valid"].item(), 0)
                self.assertTrue(torch.isfinite(pair["bbox"]).all())
                self.assertEqual(tuple(pair["person"].shape), (3, 224, 224))

    def test_shared_flip_preserves_person_orientation(self):
        transform = PairedFullPersonTransform(True, normalization="none", margin=0,
                                             jitter_probability=0)
        image = Image.new("RGB", (100, 100))
        image.paste((255, 0, 0), (0, 0, 50, 100))
        with patch("torchvision.transforms.RandomResizedCrop.get_params",
                   return_value=(0, 0, 100, 100)), patch("torch.rand", return_value=torch.tensor([0.0])):
            pair = transform(image, [0, 0, 100, 100])
        self.assertGreater(pair["full"][0, 112, 200], .9)
        self.assertGreater(pair["person"][0, 112, 200], .9)
        self.assertLess(pair["person"][0, 112, 20], .1)

    def test_two_people_in_same_image_keep_target_and_id_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (100, 100))
            image.paste((255, 0, 0), (0, 0, 50, 100))
            path = Path(directory) / "two_people.png"
            image.save(path)
            source = object.__new__(EMOTIC)
            source.file_paths = [str(path), str(path)]
            source.body_bboxes = [[0, 0, 50, 100], [50, 0, 100, 100]]
            source.targets = [[0], [1]]
            source.classes = ["a", "b"]
            source.sample_ids = ["val:x#person=0", "val:x#person=1"]
            source.paired_transform = PairedFullPersonTransform(False, normalization="none", margin=0)
            view = LabelView(source, [1, 0], [0, 1], include_sample_id=True)
            pairs, targets, ids = next(iter(DataLoader(view, batch_size=2)))
            self.assertEqual(list(ids), ["val:x#person=1", "val:x#person=0"])
            self.assertTrue(torch.equal(targets, torch.tensor([[0., 1.], [1., 0.]])))
            self.assertTrue(torch.equal(pairs["full"][0], pairs["full"][1]))
            self.assertGreater(pairs["person"][1].sum(), pairs["person"][0].sum())


if __name__ == "__main__":
    unittest.main()
