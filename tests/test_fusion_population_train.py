"""Protocol tests for TRAIN population changes beside one branch trajectory.

The synthetic images deliberately contain current, historical and future-only
people together. Image grouping alone must not grant access to future people.
"""
import copy
import random
import unittest

import numpy as np
import torch

from multi_lane.track_a import runner
from multi_lane.track_a.fusion_population import (
    DualPopulationFusion, population_indices, population_manifest,
)
from scripts.emotic.run_r1_fusion_return_train import ReturnFusion


METHODS = ("current_positive", "seen_population")


class MixedPeople:
    def __init__(self):
        self.sample_ids, self.targets, self.face_training_valid = [], [], []
        # In task1, class5 is current, class0 historical, class23 future.
        for image in range(100):
            for person, labels in enumerate(([5], [0], [23], [5, 23], [])):
                self.sample_ids.append(f"train:population_image{image}#person={person}")
                self.targets.append(list(labels))
                self.face_training_valid.append(person != 1)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        labels = torch.zeros(26)
        labels[self.targets[index]] = 1
        return torch.tensor([float(index)]), labels


def image_id(name):
    return str(name).rsplit("#person=", 1)[0]


def observations(classes=3, seed=193):
    gen = torch.Generator().manual_seed(seed)
    ids = [f"train:current{i}#person=0" for i in range(8)]
    z = torch.randn(8, 3, classes, generator=gen, requires_grad=True)
    y = torch.zeros(8, classes)
    y[torch.arange(8), torch.arange(8) % classes] = 1
    reliable = torch.tensor([True, False] * 4)
    negative_ids = [f"train:historical{i}#person=0" for i in range(6)]
    negative_z = torch.randn(6, 3, classes, generator=gen, requires_grad=True)
    combined = (
        ids + negative_ids,
        torch.cat((z, negative_z)),
        torch.cat((y, torch.zeros(6, classes))),
        torch.cat((reliable, torch.tensor([False, True] * 3))),
    )
    return {"current_positive": (ids, z, y, reliable), "seen_population": combined}


class EarlyReturnProtocolTests(unittest.TestCase):
    def test_device_ordinals_follow_parent_visibility(self):
        from scripts.emotic.run_r1_fusion_population_train import branch_gpu_map, phase_sizes
        self.assertEqual(branch_gpu_map("0,1,0", "1,0"),
                         {"full": "1", "person": "0", "face": "1"})
        with self.assertRaises(ValueError):
            branch_gpu_map("0,1,0", "1")
        self.assertEqual(phase_sizes(False, "early_return"), (15, 15))
        self.assertEqual(phase_sizes(False), (30, 5))

    def test_expanded_only_snapshot_resumes_equivalent_heldout_fitter(self):
        data = observations()
        paired = DualPopulationFusion((3,), steps=6)
        single = DualPopulationFusion((3,), steps=6, expanded_only=True)
        for model in (paired, single):
            model.begin_task(0)
            model.fit(data)
        self.assertEqual(set(single.reports), {"seen_population"})
        self.assertEqual(single.reports["seen_population"], paired.reports["seen_population"])
        restored = DualPopulationFusion((3,), steps=6, expanded_only=True)
        restored.load_snapshot(single.snapshot())
        self.assertEqual(restored.reports, single.reports)
        self.assertEqual(restored.updates, single.updates)


class PopulationSelectionTests(unittest.TestCase):
    def test_exact_original_positive_holdout_and_only_seen_negative_people(self):
        source = MixedPeople()
        parts = population_indices(source, 1)
        a, h = runner.fit_calibration_indices(source, runner.task_indices(1), .2)
        self.assertEqual(parts["current_positive"], h)
        self.assertTrue(parts["added_negative"])
        self.assertEqual(parts["seen_population"], h + parts["added_negative"])
        self.assertEqual(len(parts["seen_population"]), len(set(parts["seen_population"])))
        for index in parts["added_negative"]:
            self.assertEqual(source.targets[index], [0])
            # Invalid Face is still a valid two-view fusion example.
            self.assertFalse(source.face_training_valid[index])
        self.assertFalse({image_id(source.sample_ids[i]) for i in a} &
                         {image_id(source.sample_ids[i]) for i in parts["seen_population"]})
        self.assertTrue(all(source.targets[i] for i in parts["seen_population"]))
        self.assertFalse(any(source.targets[i] == [23] for i in parts["seen_population"]))
        # A current person with an additional future annotation is legal;
        # the future target must subsequently be absent from supervision.
        self.assertTrue(any(source.targets[i] == [5, 23] for i in h))

    def test_labels_are_current_task_only_and_background_remains_all_zero(self):
        source = MixedPeople()
        parts = population_indices(source, 1)
        data = runner.LabelView(source, parts["seen_population"], runner.task_indices(1))
        for row in range(len(data)):
            _, y = data[row]
            self.assertEqual(tuple(y.shape), (3,))
            self.assertEqual(float(y.sum()), 1. if row < len(parts["current_positive"]) else 0.)

    def test_task0_has_no_historical_people_to_add(self):
        source = MixedPeople()
        parts = population_indices(source, 0)
        self.assertEqual(parts["added_negative"], [])
        self.assertEqual(parts["current_positive"], parts["seen_population"])

    def test_population_is_view_independent_and_deterministic(self):
        source = MixedPeople()
        expected = population_indices(source, 1)
        other = copy.deepcopy(source)
        other.face_training_valid = [True] * len(other)
        self.assertEqual(expected, population_indices(other, 1))
        self.assertEqual(expected, population_indices(source, 1))

    def test_official_evaluation_and_duplicate_ids_are_rejected(self):
        for bad in ("val", "test", "duplicate"):
            source = MixedPeople()
            if bad == "duplicate":
                source.sample_ids[-1] = source.sample_ids[0]
            else:
                source.sample_ids = [name.replace("train:", bad+":") for name in source.sample_ids]
            with self.assertRaises(ValueError):
                population_indices(source, 1)

    def test_manifest_accounts_for_historical_access_and_no_branch_replay(self):
        source = MixedPeople()
        selection = population_indices(source, 1)
        prior = {image_id(source.sample_ids[i]) for i in selection["added_negative"]}
        manifest = population_manifest(source, 1, prior_returned_groups=prior)
        self.assertEqual(manifest["fit_split"], "train")
        self.assertTrue(manifest["historical_train_access"])
        self.assertEqual(manifest["added_historical_people"], len(selection["added_negative"]))
        self.assertEqual(manifest["added_historical_image_groups"], len(prior))
        self.assertEqual(manifest["branch_replay_updates"], 0)
        self.assertEqual(manifest["future_only_access_for_fitting"], 0)
        self.assertFalse(manifest["whole_trajectory_unseen_claim"])
        self.assertEqual(manifest["current_target_all_negative"]["seen_population"],
                         len(selection["added_negative"]))
        self.assertEqual(manifest["observation_cost"]["added_branch_forwards_total"],
                         3 * len(selection["added_negative"]))
        self.assertEqual(manifest["observation_cost"]["added_float32_logit_payload_bytes"],
                         3 * 3 * 4 * len(selection["added_negative"]))
        self.assertEqual(manifest["prior_returned_group_exposure"]["added_negative"],
                         len(selection["added_negative"]))
        first = population_manifest(source, 0)
        self.assertFalse(first["historical_train_access"])
        self.assertEqual(first["observation_cost"]["added_branch_forwards_total"], 0)


class PairedFusionTests(unittest.TestCase):
    def assertNestedEqual(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assertNestedEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertNestedEqual(a, b)
        else:
            self.assertEqual(left, right)

    def test_equal_initialization_and_exact_formal_budget_for_both_populations(self):
        dual = DualPopulationFusion((3,))
        dual.begin_task(0)
        for fitter in dual.banks.values():
            weights = fitter.bank.tasks[0].weight_matrix()
            torch.testing.assert_close(weights[0], torch.tensor([.5, .5, 0.]))
            torch.testing.assert_close(weights[1], torch.tensor([1/3, 1/3, 1/3]))
        reports = dual.fit(observations())
        for method in METHODS:
            self.assertEqual(dual.updates[method][0], 100)
            self.assertEqual(reports[method]["steps"], 100)
            self.assertEqual(len(reports[method]["trajectory"]), 100)
            for state in dual.snapshot()["banks"][method]["optimizers"][0]["state"].values():
                self.assertEqual(int(state["step"]), 100)

    def test_positive_control_reproduces_original_fitter_exactly(self):
        rows = observations()
        dual = DualPopulationFusion((3,), steps=4)
        original = ReturnFusion((3,), steps=4)
        dual.begin_task(0)
        original.begin_task(0)
        dual.fit(rows)
        original.fit(*rows["current_positive"])
        self.assertNestedEqual(dual.banks["current_positive"].snapshot(), original.snapshot())

    def test_no_extra_people_means_identical_banks(self):
        rows = observations()
        rows["seen_population"] = rows["current_positive"]
        dual = DualPopulationFusion((3,), steps=3)
        dual.begin_task(0)
        dual.fit(rows)
        self.assertNestedEqual(dual.banks[METHODS[0]].snapshot(), dual.banks[METHODS[1]].snapshot())

    def test_live_branch_logits_are_detached_and_random_streams_unchanged(self):
        rows = observations()
        original_logits = rows["current_positive"][1]
        before_logits = original_logits.detach().clone()
        dual = DualPopulationFusion((3,), steps=3)
        dual.begin_task(0)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        dual.fit(rows)
        self.assertIsNone(original_logits.grad)
        self.assertTrue(torch.equal(original_logits, before_logits))
        self.assertEqual(py_state, random.getstate())
        after_np = np.random.get_state()
        self.assertEqual(np_state[0], after_np[0])
        self.assertTrue(np.array_equal(np_state[1], after_np[1]))
        self.assertEqual(np_state[2:], after_np[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_both_inputs_validate_before_any_bank_updates(self):
        for invalid in ("official_test", "duplicate", "nonfinite", "float_mask", "added_positive"):
            rows = observations()
            ids, z, y, mask = rows["seen_population"]
            ids, z, y, mask = list(ids), z.detach().clone(), y.clone(), mask.clone()
            if invalid == "official_test":
                ids[-1] = "test:unavailable#person=0"
            elif invalid == "duplicate":
                ids[-1] = ids[0]
            elif invalid == "nonfinite":
                z[-1, 1, 0] = float("nan")
            elif invalid == "float_mask":
                mask = mask.float()
            else:
                y[-1, 0] = 1.
            rows["seen_population"] = (ids, z, y, mask)
            dual = DualPopulationFusion((3,), steps=2)
            dual.begin_task(0)
            before = dual.snapshot()
            with self.assertRaises((ValueError, FloatingPointError)):
                dual.fit(rows)
            self.assertNestedEqual(before, dual.snapshot())

    def test_common_person_observations_must_be_reused_exactly(self):
        for column in range(4):
            rows = observations()
            seen = list(rows["seen_population"])
            if column == 0:
                seen[0] = list(seen[0]); seen[0][0] = "train:different_person#person=0"
            else:
                seen[column] = seen[column].detach().clone()
                if column == 1:
                    seen[column][0, 0, 0] += .1
                elif column == 2:
                    seen[column][0, 0] = 0
                else:
                    seen[column][0] = ~seen[column][0]
            rows["seen_population"] = tuple(seen)
            dual = DualPopulationFusion((3,), steps=2)
            dual.begin_task(0)
            before = dual.snapshot()
            with self.assertRaises(ValueError):
                dual.fit(rows)
            self.assertNestedEqual(before, dual.snapshot())

    def test_both_banks_exclude_unreliable_face_exactly(self):
        rows = observations()
        dual = DualPopulationFusion((3,), steps=3)
        dual.begin_task(0)
        dual.fit(rows)
        _, z, _, mask = rows["seen_population"]
        with torch.no_grad():
            other = z.clone()
            other[~mask, 2] = 1e6
            for fitter in dual.banks.values():
                self.assertTrue(torch.equal(fitter.bank(z, mask, 0)[~mask],
                                            fitter.bank(other, mask, 0)[~mask]))

    def test_old_banks_seal_and_resume_preserves_both_trajectories(self):
        dual = DualPopulationFusion((3, 2), steps=3)
        dual.begin_task(0)
        dual.fit(observations())
        snapshot = dual.snapshot()
        old = {m: b.bank.tasks[0].raw_shares.detach().clone() for m, b in dual.banks.items()}
        restored = DualPopulationFusion((3, 2), steps=3)
        restored.load_snapshot(snapshot)
        self.assertNestedEqual(snapshot, restored.snapshot())
        for model in (dual, restored):
            model.begin_task(1)
            model.fit(observations(classes=2))
            for method, fitter in model.banks.items():
                self.assertTrue(torch.equal(old[method], fitter.bank.tasks[0].raw_shares))
                self.assertFalse(fitter.bank.tasks[0].raw_shares.requires_grad)
                self.assertFalse(fitter.bank.tasks[1].raw_shares.requires_grad)
        self.assertNestedEqual(dual.snapshot(), restored.snapshot())
        # The returned snapshot is independent of subsequent model updates.
        self.assertEqual(snapshot["task"], 0)
        self.assertEqual(set(snapshot["banks"][METHODS[0]]["updates"]), {0})


if __name__ == "__main__":
    unittest.main()
