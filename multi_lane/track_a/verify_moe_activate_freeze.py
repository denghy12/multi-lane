"""Verify recorded AF task boundaries against actual compact checkpoint tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .export_compact_test_scores import _sha256, _torch_load


def read(path):
    return json.loads(path.read_text())


def verify(root, expected_tasks, baseline=None):
    train = root / "train"
    config = read(train / "config.json")
    summary = read(train / "seed_summary.json")
    protocol = read(train / "paired_protocol_audit.json")
    audit = read(train / "activate_freeze_audit.json")
    if (summary["status"] != "complete" or summary["config"] != config
        or config["git"]["dirty"] or config["max_tasks"] != expected_tasks
        or config["adapter_mode"] != "moe_task_ffn_activate_freeze"
        or not protocol["frozen_visual_unchanged"]):
        raise AssertionError("Incomplete or unpaired AF provenance")
    if baseline:
        reference = read(baseline / "paired_protocol_audit.json")
        for key in ("initial_selector_prompt_classifier_sha256", "frozen_visual_sha256_before"):
            assert protocol[key] == reference[key], key
        for task in range(expected_tasks):
            left, right = protocol["tasks"][str(task)], reference["tasks"][str(task)]
            for key in ("train_sample_ids_sha256", "first_train_batch_ids_sha256", "validation_sample_ids_sha256", "sampler_seed"):
                assert left[key] == right[key], (task, key)
        reference_config = read(baseline / "config.json")
        for key in ("seed", "train_batch_size", "eval_batch_size",
                    "optimizer", "learning_rate", "temperature", "training_loss_mode",
                    "view_fusion", "view_classifier_mode", "view_auxiliary_loss_weight",
                    "selector_mode", "prompt_mode", "input_normalization", "train_crop_scale"):
            assert config[key] == reference_config[key], key
        if config["training_budget_mode"] == "epochs":
            assert config["epochs_per_task"] == reference_config["epochs_per_task"]
        else:
            # A two-task smoke deliberately uses updates, not the formal epoch budget.
            assert expected_tasks < 8 and config["optimizer_updates_per_task"] > 0
        if expected_tasks == 8:
            assert config["training_budget_mode"] == "epochs"
    history = read(train / "training_history.json")
    checks, previous = [], None
    for task in range(expected_tasks):
        path = train / "compact_checkpoints" / f"task{task}.pth"
        payload = _torch_load(path)
        assert payload["source_git"] == config["git"] and payload["task_id"] == task
        state = payload["model"]
        frozen = state["adapter_bank.frozen_experts"]
        finalized = state["adapter_bank.finalized_tasks"]
        assert bool(finalized[:task+1].all()) and not bool(finalized[task+1:].any())
        row = audit["tasks"][str(task)]
        assert row["frozen_experts_and_old_routers_unchanged"]
        assert row["protected_sha256_before"] == row["protected_sha256_after"]
        for layer, block in enumerate(config["adapter_layer_indices"]):
            info = row["freeze"][str(block)]
            counts = state["adapter_bank.training_counts"][task, layer]
            samples = state["adapter_bank.training_samples"][task, layer]
            total = counts.sum(0)
            selected = sorted(range(config["moe_num_experts"]), key=lambda i: (-int(total[i]), i))[:2]
            assert info["top2"] == selected
            assert info["counts"] == total.tolist()
            assert info["per_view_counts"] == counts.tolist()
            assert info["per_view_samples"] == samples.tolist()
            assert bool(((counts.sum(1) >= samples) & (counts.sum(1) <= 2*samples)).all())
            assert int(samples[0]) == int(samples[1]) > 0 and int(samples[2]) <= int(samples[0])
            assert state["adapter_bank.freeze_top2"][task, layer].tolist() == selected
            expected = previous["adapter_bank.frozen_experts"][layer].clone() if previous else torch.zeros_like(frozen[layer])
            expected[selected] = True
            assert torch.equal(frozen[layer], expected)
            assert torch.equal(state["adapter_bank.frozen_history"][task, layer], expected)
            assert info["cumulative_frozen"] == torch.nonzero(expected).flatten().tolist()
        if previous:
            for layer, mask in enumerate(previous["adapter_bank.frozen_experts"]):
                for expert in torch.nonzero(mask).flatten().tolist():
                    prefix = f"adapter_bank.experts.{layer}.{expert}."
                    for name, value in previous.items():
                        if name.startswith(prefix):
                            assert torch.equal(value, state[name]), name
            for old in range(task):
                prefix = f"adapter_bank.tasks.{old}."
                for name, value in previous.items():
                    if name.startswith(prefix):
                        assert torch.equal(value, state[name]), name
        updates = sum(int(x["optimizer_steps"]) for x in history[str(task)])
        skipped = sum(int(x["skipped_optimizer_steps"]) for x in history[str(task)])
        assert updates > 0 and skipped == 0, (task, updates, skipped)
        checks.append(dict(task=task, checkpoint_sha256=_sha256(path), updates=updates,
                           skipped=skipped, frozen_count=frozen.sum(1).tolist()))
        previous = state
    if (root / "replay_val/result.json").exists():
        replay = read(root / "replay_val/result.json")
        for original, restored in zip(summary["task_metrics"], replay["task_metrics"]):
            assert abs(original["mAP"]-restored["mAP"]) < 1e-7
    result = dict(status="verified", expected_tasks=expected_tasks, checks=checks,
                  baseline_provenance_matched=str(baseline) if baseline else None,
                  source_git=config["git"], freeze_policy=config["moe_freeze_policy"])
    (root / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=8)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    verify(args.root, args.expected_tasks, args.baseline)
