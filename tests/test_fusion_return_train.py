"""Contracts for one-trajectory temporary TRAIN holdout and return.

These tests check data exposure and state preservation, rather than reproducing
an implementation's numerical expression.
"""
import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a import runner
from multi_lane.track_a.fusion_return_training import validate_return_training, reset_return_scheduler
from scripts.emotic.run_r1_fusion_return_train import ReturnFusion, phase_sizes, full_loader_like


class SyntheticPeople:
    """Several people per image; some images recur across current-task labels."""
    def __init__(self):
        self.sample_ids = [f"train:image{i}#person={p}" for i in range(80) for p in range(2)]
        self.targets = [[0, 5] if i % 3 == 0 else [0] for i in range(len(self.sample_ids))]
        self.face_training_valid = [i % 4 != 0 for i in range(len(self.sample_ids))]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        y = torch.zeros(26)
        y[self.targets[index]] = 1
        return torch.tensor([float(index)]), y


class ReturnDataTests(unittest.TestCase):
    def test_holdout_is_by_original_image_across_people_views_and_tasks(self):
        source = SyntheticPeople()
        fit0, held0 = runner.fit_calibration_indices(source, (0,), .2)
        fit1, held1 = runner.fit_calibration_indices(source, (5,), .2)
        key = lambda i: source.sample_ids[i].rsplit("#person=", 1)[0]
        self.assertTrue(fit0 and held0 and fit1 and held1)
        self.assertFalse({key(i) for i in fit0} & {key(i) for i in held0})
        self.assertEqual(set(fit0) | set(held0), set(range(len(source))))
        self.assertTrue(set(held1).issubset(set(held0)))
        self.assertTrue(set(fit1).issubset(set(fit0)))
        # Returning all legal task0 examples means task1's heldout pictures
        # have prior exposure. A split itself does not erase that exposure.
        prior_images = {key(i) for i in fit0 + held0}
        self.assertTrue(all(key(i) in prior_images for i in held1))
        clone = copy.deepcopy(source)
        self.assertEqual((fit0, held0), runner.fit_calibration_indices(clone, (0,), .2))

    def test_return_loader_adds_all_current_task_people_with_same_source(self):
        source = SyntheticPeople()
        fit, held = runner.fit_calibration_indices(source, (5,), .2)
        loader = DataLoader(runner.LabelView(source, fit, runner.task_indices(1)), batch_size=7,
                            shuffle=True, num_workers=0, drop_last=False)
        returned = full_loader_like(loader, "person", 1, False)
        expected = {i for i, target in enumerate(source.targets) if 5 in target}
        self.assertIs(returned.dataset.source, source)
        self.assertEqual(set(returned.dataset.indices), expected)
        self.assertTrue(set(held).issubset(set(returned.dataset.indices)))
        self.assertEqual(returned.dataset.class_indices, runner.task_indices(1))
        self.assertEqual(returned.batch_size, 7)
        self.assertFalse(returned.drop_last)

    def test_return_loader_preserves_face_filter_and_does_not_add_other_classes(self):
        source = SyntheticPeople()
        fit, held = runner.fit_calibration_indices(source, (5,), .2)
        fit = runner.filter_face_training_indices(source, fit)
        loader = DataLoader(runner.LabelView(source, fit, runner.task_indices(1)), batch_size=8,
                            shuffle=True, num_workers=0)
        returned = full_loader_like(loader, "face", 1, False)
        expected = {i for i, target in enumerate(source.targets)
                    if 5 in target and source.face_training_valid[i]}
        self.assertEqual(set(returned.dataset.indices), expected)
        self.assertTrue(set(i for i in held if source.face_training_valid[i]) <= expected)


class ReturnOptimizerTests(unittest.TestCase):
    def config(self):
        return dict(initial_epochs=2, return_epochs=3, lr_factor=.1, loader=object())

    def make_optimizer(self):
        a = torch.nn.Parameter(torch.tensor([1., -2.]))
        b = torch.nn.Parameter(torch.tensor([.3]))
        opt = torch.optim.Adam([{'params': [a], 'lr': .0125},
                                {'params': [b], 'lr': .0004}], weight_decay=0.)
        return (a, b), opt

    def step(self, params, opt, scheduler):
        opt.zero_grad(set_to_none=True)
        loss = sum((p*p).sum() for p in params)
        loss.backward(); opt.step(); scheduler.step()

    def test_phase_validation_rejects_ambiguous_budgets(self):
        config = self.config()
        validate_return_training(config, 5, None)
        for bad, epochs, updates in [({**config, 'return_epochs': 0}, 2, None),
                                     (config, 6, None), (config, 5, 8),
                                     ({**config, 'lr_factor': 0}, 5, None),
                                     ({**config, 'loader': None}, 5, None)]:
            with self.assertRaises(ValueError):
                validate_return_training(bad, epochs, updates)

    def test_lr_switch_preserves_adam_moments_and_parameter_identity(self):
        params, optimizer = self.make_optimizer()
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        for _ in range(2): self.step(params, optimizer, schedule)
        moments = [{k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state[p].items()}
                   for p in params]
        moment_ids = [id(optimizer.state[p]['exp_avg']) for p in params]
        new_schedule = reset_return_scheduler(optimizer, [.0125, .0004], self.config())
        self.assertIs(new_schedule.optimizer, optimizer)
        for i, p in enumerate(params):
            self.assertEqual(moment_ids[i], id(optimizer.state[p]['exp_avg']))
            for k, val in moments[i].items():
                self.assertTrue(torch.equal(val, optimizer.state[p][k]))
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .00125)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], .00004)
        self.assertEqual(new_schedule.base_lrs, [.00125, .00004])
        self.step(params, optimizer, new_schedule)
        self.assertEqual(int(optimizer.state[params[0]]['step']), 3)

    def test_mid_return_resume_matches_uninterrupted_optimizer(self):
        params, optimizer = self.make_optimizer()
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        for _ in range(2): self.step(params, optimizer, schedule)
        schedule = reset_return_scheduler(optimizer, [.0125, .0004], self.config())
        self.step(params, optimizer, schedule)
        state = copy.deepcopy(optimizer.state_dict())
        sched_state = copy.deepcopy(schedule.state_dict())
        saved_params = [p.detach().clone() for p in params]
        other_params, other_optimizer = self.make_optimizer()
        for p, value in zip(other_params, saved_params): p.data.copy_(value)
        # This is the runner's restore order: construct the correct phase
        # schedule, then restore optimizer and scheduler state.
        other_schedule = reset_return_scheduler(other_optimizer, [.0125, .0004], self.config())
        other_optimizer.load_state_dict(state)
        other_schedule.load_state_dict(sched_state)
        for _ in range(2):
            self.step(params, optimizer, schedule)
            self.step(other_params, other_optimizer, other_schedule)
        for p, q in zip(params, other_params):
            self.assertTrue(torch.equal(p, q))
            for key, val in optimizer.state[p].items():
                self.assertTrue(torch.equal(val, other_optimizer.state[q][key]))
        self.assertEqual(schedule.state_dict(), other_schedule.state_dict())


class ReturnFusionTests(unittest.TestCase):
    def rows(self, classes=2):
        rng = torch.Generator().manual_seed(711)
        return ([f"train:held{i}#person=0" for i in range(10)],
                torch.randn(10, 3, classes, generator=rng, requires_grad=True),
                torch.randint(0, 2, (10, classes), generator=rng).float(),
                torch.tensor([True, False] * 5))

    def test_phase_budget_is_predeclared(self):
        self.assertEqual(phase_sizes(False), (30, 5))
        self.assertEqual(phase_sizes(True), (2, 1))

    def test_formal_fit_completes_exactly_one_hundred_updates(self):
        fusion = ReturnFusion((2,))
        fusion.begin_task(0)
        report = fusion.fit(*self.rows())
        self.assertEqual(fusion.updates[0], 100)
        self.assertEqual(report['steps'], 100)
        self.assertEqual(len(report['trajectory']), 100)
        self.assertEqual(report['trajectory'][-1]['step'], 100)
        for value in fusion.snapshot()['optimizers'][0]['state'].values():
            self.assertEqual(int(value['step']), 100)

    def test_fit_detaches_branches_and_seals_gate(self):
        fusion = ReturnFusion((2,), steps=2)
        fusion.begin_task(0)
        before = fusion.bank.tasks[0].raw_shares.detach().clone()
        rows = self.rows()
        fusion.fit(*rows)
        self.assertIsNone(rows[1].grad)
        self.assertFalse(torch.equal(before, fusion.bank.tasks[0].raw_shares))
        self.assertFalse(fusion.bank.tasks[0].raw_shares.requires_grad)
        with self.assertRaises((ValueError, RuntimeError)):
            fusion.fit(*rows)

    def test_no_official_validation_or_test_can_update(self):
        for split in ('val', 'test'):
            fusion = ReturnFusion((2,), steps=2)
            fusion.begin_task(0)
            ids, *rest = self.rows()
            before = fusion.bank.tasks[0].raw_shares.detach().clone()
            with self.assertRaises(ValueError):
                fusion.fit([name.replace('train:', split + ':') for name in ids], *rest)
            self.assertTrue(torch.equal(before, fusion.bank.tasks[0].raw_shares))

    def test_unreliable_face_contributes_exactly_zero(self):
        fusion = ReturnFusion((2,), steps=2)
        fusion.begin_task(0)
        _, z, y, mask = rows = self.rows()
        fusion.fit(*rows)
        with torch.no_grad():
            pred = fusion.bank(z, mask, 0)
            other = z.clone()
            other[~mask, 2] += 1000
            self.assertTrue(torch.equal(pred[~mask], fusion.bank(other, mask, 0)[~mask]))

    def test_old_task_and_snapshot_remain_unchanged(self):
        fusion = ReturnFusion((2, 1), steps=2)
        fusion.begin_task(0)
        fusion.fit(*self.rows())
        snapshot = fusion.snapshot()
        old = fusion.bank.tasks[0].raw_shares.detach().clone()
        fusion.begin_task(1)
        fusion.fit(*self.rows(1))
        self.assertTrue(torch.equal(old, fusion.bank.tasks[0].raw_shares))
        self.assertFalse(fusion.bank.tasks[0].raw_shares.requires_grad)
        restored = ReturnFusion((2, 1), steps=2)
        restored.load_snapshot(snapshot)
        self.assertTrue(torch.equal(old, restored.bank.tasks[0].raw_shares))
        restored.begin_task(1)
        restored.fit(*self.rows(1))
        for name, value in fusion.bank.state_dict().items():
            self.assertTrue(torch.equal(value, restored.bank.state_dict()[name]), name)

    def test_invalid_observations_are_rejected(self):
        for invalid in ('duplicate', 'nonfinite', 'shape', 'float_mask', 'nonbinary'):
            fusion = ReturnFusion((2,), steps=2)
            fusion.begin_task(0)
            ids, z, y, mask = self.rows()
            if invalid == 'duplicate':
                ids[-1] = ids[0]
            elif invalid == 'nonfinite':
                z = z.detach().clone(); z[0, 0, 0] = float('nan')
            elif invalid == 'float_mask':
                mask = mask.float()
            elif invalid == 'nonbinary':
                y[0, 0] = .5
            else:
                y = y[:, :1]
            with self.assertRaises((ValueError, FloatingPointError)):
                fusion.fit(ids, z, y, mask)


if __name__ == '__main__':
    unittest.main()
