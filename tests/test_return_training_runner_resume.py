"""Exercise the actual training loop across and after the loader/LR boundary."""
import copy
import unittest

import torch
from torch import nn

from multi_lane.track_a import runner


class TinyTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.selectors = nn.Parameter(torch.tensor([0.1, -0.1]))
        self.prompts = nn.ParameterList()
        self.head = nn.Linear(2, 5)
        self.selector_conditioner = None
        self.snapshots = {}

    def base_optimizer_parameters(self):
        return self.parameters()

    def conditioning_optimizer_parameters(self):
        return []

    def adapter_optimizer_parameters(self):
        return []

    def current_all_logits(self, inputs):
        return self.head(inputs + self.selectors)

    current_logits = current_all_logits

    def save_training_epoch(self, task, epoch, history, optimizer, scheduler, scaler, updates):
        self.snapshots[epoch] = {
            "model": copy.deepcopy(self.state_dict()),
            "training": copy.deepcopy({
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "history": history,
                "next_epoch": epoch, "completed_task_updates": updates,
            }),
        }


def assert_nested_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for key in a:
            assert_nested_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class RunnerReturnResumeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.initial = [(torch.tensor([[1., 0.], [0., 1.]]),
                         torch.tensor([[1., 0., 1., 0., 1.], [0., 1., 0., 1., 0.]]))]
        self.returned = self.initial + [
            (torch.tensor([[2., 1.], [-1., 2.]]),
             torch.tensor([[0., 1., 0., 0., 1.], [1., 0., 1., 1., 0.]]))]
        self.origin = copy.deepcopy(TinyTask().state_dict())

    def run_training(self, initial_state, resume=None, phases=True, epochs=5, restart_scheduler=True):
        model = TinyTask()
        model.load_state_dict(initial_state)
        if phases:
            model.r1_return_training = dict(initial_epochs=2, return_epochs=3,
                lr_factor=.1 if restart_scheduler else 1., loader=self.returned,
                restart_scheduler=restart_scheduler)
        if resume is not None:
            model.r1_resume = copy.deepcopy(resume)
        runner.train_task(model, self.initial, self.returned, torch.device("cpu"),
            task_id=0, epochs=epochs, learning_rate=.01, weight_decay=0,
            temperature=1, amp=False, loss_mode="current_only")
        return model

    def test_actual_loop_resume_exact_at_and_after_boundary(self):
        whole = self.run_training(self.origin)
        for cutoff in (2, 3):
            with self.subTest(cutoff=cutoff):
                saved = whole.snapshots[cutoff]
                resumed = self.run_training(saved["model"], saved["training"])
                assert_nested_equal(self, whole.state_dict(), resumed.state_dict())
                for key in ("optimizer", "scheduler", "scaler", "completed_task_updates"):
                    assert_nested_equal(self, whole.snapshots[5]["training"][key],
                        resumed.snapshots[5]["training"][key])
        rows = whole.snapshots[5]["training"]["history"]
        self.assertEqual([r["optimizer_steps"] for r in rows], [1, 1, 2, 2, 2])
        for observed, expected in zip([r["learning_rate"] for r in rows], [.01, .005, .001, .00075, .00025]):
            self.assertAlmostEqual(observed, expected, places=12)
        self.assertEqual([r["data_phase"] for r in rows],
            ["initial_train", "initial_train", "return_full_train", "return_full_train", "return_full_train"])
        self.assertEqual(whole.snapshots[2]["training"]["scheduler"]["T_max"], 2)
        self.assertEqual(whole.snapshots[5]["training"]["completed_task_updates"], 8)

    def test_continuous_cosine_uses_original_schedule_and_restores_at_boundary(self):
        staged = self.run_training(self.origin, restart_scheduler=False)
        default = self.run_training(self.origin, phases=False)
        last = staged.snapshots[5]["training"]
        for epoch in (1, 2):
            assert_nested_equal(self, staged.snapshots[epoch]["model"], default.snapshots[epoch]["model"])
        self.assertEqual(last["scheduler"], default.snapshots[5]["training"]["scheduler"])
        self.assertEqual([r["learning_rate"] for r in last["history"]],
                         [r["learning_rate"] for r in default.snapshots[5]["training"]["history"]])
        self.assertEqual([r["optimizer_steps"] for r in last["history"]], [1, 1, 2, 2, 2])
        for cutoff in (2, 3):
            saved = staged.snapshots[cutoff]
            resumed = self.run_training(saved["model"], saved["training"], restart_scheduler=False)
            assert_nested_equal(self, staged.state_dict(), resumed.state_dict())
            for key in ("optimizer", "scheduler", "scaler", "completed_task_updates"):
                assert_nested_equal(self, last[key], resumed.snapshots[5]["training"][key])

    def test_initial_phase_exactly_matches_unmodified_default_loop(self):
        staged = self.run_training(self.origin)
        original = self.run_training(self.origin, phases=False, epochs=2)
        assert_nested_equal(self, staged.snapshots[2]["model"], original.state_dict())
        for key in ("optimizer", "scheduler", "scaler", "completed_task_updates"):
            assert_nested_equal(self, staged.snapshots[2]["training"][key],
                original.snapshots[2]["training"][key])


if __name__ == "__main__":
    unittest.main()
