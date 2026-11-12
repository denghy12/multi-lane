"""Fusion schedules and observation isolation for one ongoing three-view model."""
from contextlib import contextmanager
import copy

import numpy as np
import torch
from torch.nn import functional as F

from .r1_crossvit import rng_state, restore_rng
from .task_logit_fusion import TaskLogitFusion

METHODS = ("fixed", "continuous", "mid_freeze")


def update_budget(method, epoch, smoke=False):
    if method == "fixed":
        return 0
    if method not in METHODS:
        raise ValueError("Unknown schedule")
    if smoke:
        return (1 if epoch in (2, 3) else 0) if method == "continuous" else (2 if epoch == 2 else 0)
    return (4 if 6 <= epoch <= 30 else 0) if method == "continuous" else (10 if 6 <= epoch <= 15 else 0)


@contextmanager
def isolated_observation(model):
    """Restore Python/NumPy/Torch RNG, threads, modes and gradient flags exactly."""
    rng = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    grad_flags = [(p, p.requires_grad) for p in model.parameters()]
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        for p, flag in grad_flags:
            p.requires_grad_(flag)
        torch.set_num_threads(threads)
        restore_rng(rng)


class OnlineFusionSchedules:
    """Two predeclared readouts optimized beside the same protected branch path."""
    def __init__(self, task_sizes, device="cpu", smoke=False):
        self.banks = {name: TaskLogitFusion(task_sizes).to(device) for name in METHODS[1:]}
        self.optimizers = {}
        self.optimizer_states = {name: {} for name in self.banks}
        self.updates = {name: 0 for name in self.banks}
        self.task = -1
        self.last_epoch = 0
        self.device, self.smoke = device, smoke

    def begin_task(self, task):
        if task != self.task + 1:
            raise ValueError("Tasks must be consecutive")
        self.task, self.last_epoch = task, 0
        for name, bank in self.banks.items():
            gate = bank.add_task(task).to(self.device)
            self.optimizers[name] = torch.optim.Adam(gate.parameters(), lr=.02, weight_decay=0)
            self.updates[name] = 0

    def observe(self, epoch, ids, logits, targets, reliable):
        if epoch != self.last_epoch + 1:
            raise ValueError("Epoch observations must be consecutive")
        self.last_epoch = epoch
        if any(not str(s).startswith("train:") for s in ids):
            raise ValueError("Only TRAIN may update fusion")
        n, classes = len(ids), self.banks["continuous"].task_sizes[self.task]
        if (len(set(ids)) != n or tuple(logits.shape) != (n, 3, classes)
                or tuple(targets.shape) != (n, classes) or tuple(reliable.shape) != (n,)):
            raise ValueError("Misaligned current-task TRAIN observations")
        # Deliberately detach even if a future caller supplies live tensors.
        z = torch.as_tensor(logits, dtype=torch.float32, device=self.device).detach()
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device).detach()
        good = torch.as_tensor(reliable, dtype=torch.bool, device=self.device)
        if not torch.isfinite(z).all() or not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite fusion inputs")
        reports = {}
        for name, bank in self.banks.items():
            old = {k: v.clone() for k, v in bank.state_dict().items()
                   if not k.startswith(f"tasks.{self.task}.")}
            gate, opt = bank.tasks[self.task], self.optimizers[name]
            steps = update_budget(name, epoch, self.smoke)
            gate.requires_grad_(bool(steps))
            before = float(F.binary_cross_entropy_with_logits(gate(z, good), y).detach())
            gradients = []
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(gate(z, good), y)
                loss.backward()
                g = gate.raw_shares.grad
                if not torch.isfinite(loss) or g is None or not torch.isfinite(g).all():
                    raise FloatingPointError("Invalid fusion objective/gradient")
                gradients.append(float(g.norm()))
                opt.step()
            opt.zero_grad(set_to_none=True)
            self.updates[name] += steps
            if any(not torch.equal(v, bank.state_dict()[k]) for k, v in old.items()):
                raise AssertionError("An old task fusion parameter changed")
            reports[name] = {"steps": steps, "total_steps": self.updates[name], "bce_before": before,
                "bce_after": float(F.binary_cross_entropy_with_logits(gate(z, good), y).detach()),
                "gradient_norms": gradients, "weights": gate.weight_matrix().detach().cpu().tolist()}
            gate.requires_grad_(False)
            self.optimizer_states[name][self.task] = copy.deepcopy(opt.state_dict())
        return reports

    def warmup(self, epoch):
        if epoch != self.last_epoch + 1 or any(update_budget(n, epoch, self.smoke) for n in self.banks):
            raise ValueError("Not a warmup epoch")
        self.last_epoch = epoch
        return {n: {"steps": 0, "total_steps": self.updates[n],
                    "weights": b.tasks[self.task].weight_matrix().detach().cpu().tolist()}
                for n, b in self.banks.items()}

    def state_dict(self):
        return {"task": self.task, "epoch": self.last_epoch, "smoke": self.smoke,
            "banks": {n: {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
                      for n, b in self.banks.items()},
            "optimizers": copy.deepcopy(self.optimizer_states), "updates": dict(self.updates)}

    def load_state_dict(self, state):
        if self.task != -1 or state["smoke"] != self.smoke:
            raise ValueError("Restore into a fresh matching schedule")
        for task in range(state["task"] + 1):
            self.begin_task(task)
        for name, bank in self.banks.items():
            bank.load_state_dict(state["banks"][name], strict=True)
            bank.requires_grad_(False)
            self.optimizers[name].load_state_dict(state["optimizers"][name][state["task"]])
        self.optimizer_states = copy.deepcopy(state["optimizers"])
        self.updates = dict(state["updates"])
        self.last_epoch = state["epoch"]
