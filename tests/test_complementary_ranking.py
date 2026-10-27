import json
import math
import unittest

import torch
from multi_lane.track_a.complementary_ranking import (
    complementary_ranking_objective, _view_gradient_fused_logits,
)
from multi_lane.track_a.view_fusion import TaskwiseViewFusion


class ComplementaryRankingTest(unittest.TestCase):
    def fixture(self, *, one_lane=True):
        generator = torch.Generator().manual_seed(729)
        shape = (4, 1, 3) if one_lane else (4, 3)
        features = {name: torch.randn(shape, generator=generator).requires_grad_()
                    for name in ("full", "person", "face")}
        # Avoid consuming global RNG for module construction in the fixtures.
        with torch.random.fork_rng():
            torch.manual_seed(813)
            head = torch.nn.Linear(3, 4)
        reliable = torch.tensor([True, True, False, True])
        weights = torch.tensor([[.64, .16, .20], [.64, .16, .20],
                                [.80, .20, .0], [.64, .16, .20]])
        if one_lane:
            weights = weights.unsqueeze(1)
        weights.requires_grad_()
        full = features["full"][:, 0] if one_lane else features["full"]
        view_logits = {"full": head(full)}
        view_logits["full"].retain_grad()
        return dict(view_features=features, head=head, fusion_weights=weights,
                    view_logits=view_logits,
                    targets=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]]),
                    current_class_indices=(0, 2), face_reliable=reliable)

    def reference(self, args, conditioned=True, temperature=1.):
        features = args["view_features"]
        values = [features[name].detach() for name in ("full", "person", "face")]
        if values[0].ndim == 3:
            values = [value[:, 0] for value in values]
        weights = args["fusion_weights"].detach().reshape(-1, 3)
        # An independent scalar-loop oracle over natural positive/negative pairs.
        predictions = args["head"]((torch.stack(values, 1) * weights[:, :, None]).sum(1))
        full = args["view_logits"]["full"].detach()
        person_total = face_total = 0.
        pair_count = face_count = 0
        targets = args["targets"]
        for column, emotion in enumerate(args["current_class_indices"]):
            for i in range(len(targets)):
                for j in range(len(targets)):
                    if targets[i, column] != 1 or targets[j, column] != 0:
                        continue
                    margin = float(predictions[i, emotion] - predictions[j, emotion])
                    full_margin = float(full[i, emotion] - full[j, emotion])
                    difficulty = (1. + 1. / (1. + math.exp(full_margin / temperature))
                                  if conditioned else 1.)
                    value = difficulty * math.log1p(math.exp(-margin / temperature))
                    pair_count += 1
                    person_total += value
                    if args["face_reliable"][i] and args["face_reliable"][j]:
                        face_count += 1
                        face_total += value
        if not pair_count:
            return 0., 0, 0
        return (person_total + face_total) / (2 * pair_count), pair_count, face_count

    def test_matches_hard_label_pair_oracle_and_serializable_metrics(self):
        args = self.fixture()
        loss, metrics = complementary_ranking_objective(**args)
        expected, pairs, faces = self.reference(args)
        self.assertAlmostEqual(float(loss), expected, places=6)
        self.assertEqual(metrics["pair_count"], pairs)
        self.assertEqual(metrics["face_pair_count"], faces)
        self.assertEqual(metrics["extra_branch_logits_max_abs_difference"], 0.)
        self.assertEqual(metrics["active_class_count"], 2)
        self.assertGreater(metrics["mean_pair_difficulty"], 1.)
        self.assertLess(metrics["mean_pair_difficulty"], 2.)
        json.dumps(metrics, allow_nan=False)

    def test_extra_prediction_values_exactly_replay_original_feature_fusion(self):
        args = self.fixture()
        fusion = TaskwiseViewFusion(num_tasks=1, feature_dim=3, mode="fixed_three_view")
        fused, weights = fusion(args["view_features"], (0,), args["face_reliable"])
        expected = args["head"](fused)[:, 0]
        for name in ("person", "face"):
            actual = _view_gradient_fused_logits(args["view_features"], args["head"], weights, name)
            self.assertTrue(torch.equal(actual, expected))

    def test_extra_backward_updates_only_auxiliary_feature_edges(self):
        args = self.fixture()
        loss, _ = complementary_ranking_objective(**args)
        loss.backward()
        features = args["view_features"]
        self.assertIsNone(features["full"].grad)
        self.assertIsNone(args["head"].weight.grad)
        self.assertIsNone(args["head"].bias.grad)
        self.assertIsNone(args["view_logits"]["full"].grad)
        self.assertIsNone(args["fusion_weights"].grad)
        self.assertGreater(float(features["person"].grad.norm()), 0.)
        self.assertGreater(float(features["face"].grad.norm()), 0.)
        self.assertTrue(torch.equal(features["face"].grad[2], torch.zeros_like(features["face"].grad[2])))

    def test_uniform_control_has_exact_same_pairs_and_masks(self):
        args = self.fixture()
        uniform, metrics = complementary_ranking_objective(**args, full_conditioned=False)
        expected, pairs, faces = self.reference(args, conditioned=False)
        self.assertAlmostEqual(float(uniform), expected, places=6)
        self.assertEqual(metrics["pair_count"], pairs)
        self.assertEqual(metrics["face_pair_count"], faces)
        self.assertEqual(metrics["mean_pair_difficulty"], 1.)

    def test_face_loss_uses_all_pair_denominator_without_amplification(self):
        args = self.fixture()
        _, metrics = complementary_ranking_objective(**args)
        # Both branch values are equal: their loss ratio must equal the
        # oracle's masked sums, not a separate average over reliable pairs.
        expected, _, _ = self.reference(args)
        self.assertAlmostEqual((metrics["person_ranking_loss"] + metrics["face_ranking_loss"]) / 2,
                               expected, places=6)
        self.assertLess(metrics["face_ranking_loss"], metrics["person_ranking_loss"])

    def test_all_missing_faces_have_zero_signal_and_person_still_learns(self):
        args = self.fixture()
        args["face_reliable"] = torch.zeros(4, dtype=torch.bool)
        weights = args["fusion_weights"].detach().clone()
        weights[:, 0] = torch.tensor([.8, .2, .0])
        args["fusion_weights"] = weights
        loss, metrics = complementary_ranking_objective(**args)
        loss.backward()
        self.assertEqual(metrics["face_pair_count"], 0)
        self.assertEqual(metrics["face_ranking_loss"], 0.)
        self.assertTrue(torch.equal(args["view_features"]["face"].grad,
                                    torch.zeros_like(args["view_features"]["face"])))
        self.assertGreater(float(args["view_features"]["person"].grad.norm()), 0.)

    def test_no_natural_pairs_returns_connected_zero(self):
        args = self.fixture()
        args["targets"] = torch.ones_like(args["targets"])
        loss, metrics = complementary_ranking_objective(**args)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(metrics["pair_count"], 0)
        loss.backward()
        for name in ("person", "face"):
            self.assertTrue(torch.equal(args["view_features"][name].grad,
                                        torch.zeros_like(args["view_features"][name])))
        self.assertIsNone(args["view_features"]["full"].grad)
        self.assertIsNone(args["head"].weight.grad)

    def test_two_dimensional_features_match_single_lane_values_and_gradients(self):
        lane_args = self.fixture(one_lane=True)
        flat_args = self.fixture(one_lane=False)
        lane_loss, lane_metrics = complementary_ranking_objective(**lane_args)
        flat_loss, flat_metrics = complementary_ranking_objective(**flat_args)
        self.assertTrue(torch.equal(lane_loss, flat_loss))
        self.assertEqual(lane_metrics, flat_metrics)
        lane_loss.backward()
        flat_loss.backward()
        for name in ("person", "face"):
            self.assertTrue(torch.equal(lane_args["view_features"][name].grad[:, 0],
                                        flat_args["view_features"][name].grad))

    def test_reordering_people_does_not_change_objective(self):
        args = self.fixture()
        original, _ = complementary_ranking_objective(**args)
        order = torch.tensor([3, 1, 0, 2])
        args["view_features"] = {k: v[order] for k, v in args["view_features"].items()}
        args["fusion_weights"] = args["fusion_weights"][order]
        args["view_logits"] = {k: v[order] for k, v in args["view_logits"].items()}
        args["targets"] = args["targets"][order]
        args["face_reliable"] = args["face_reliable"][order]
        reordered, _ = complementary_ranking_objective(**args)
        self.assertAlmostEqual(float(original), float(reordered), places=6)

    def test_only_current_class_columns_receive_supervision(self):
        args = self.fixture()
        before, _ = complementary_ranking_objective(**args)
        with torch.no_grad():
            args["head"].weight[[1, 3]] += 10000.
            args["head"].bias[[1, 3]] -= 10000.
            args["view_logits"]["full"][:, [1, 3]] += 10000.
        after, _ = complementary_ranking_objective(**args)
        self.assertTrue(torch.equal(before, after))

    def test_does_not_mutate_parameters_features_or_rng(self):
        args = self.fixture()
        rng = torch.get_rng_state().clone()
        weights = args["head"].weight.detach().clone()
        original = {k: v.detach().clone() for k, v in args["view_features"].items()}
        complementary_ranking_objective(**args)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(weights, args["head"].weight))
        for name in original:
            self.assertTrue(torch.equal(original[name], args["view_features"][name]))

    def test_rejects_nonbinary_labels_and_invalid_temperature_or_shapes(self):
        args = self.fixture()
        for temperature in (0., -1., float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                complementary_ranking_objective(**args, temperature=temperature)
        args["targets"][0, 0] = .5
        with self.assertRaises(ValueError):
            complementary_ranking_objective(**args)
        args = self.fixture()
        args["current_class_indices"] = (0, 0)
        with self.assertRaises(ValueError):
            complementary_ranking_objective(**args)


if __name__ == "__main__":
    unittest.main()
