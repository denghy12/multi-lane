import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_lane.track_a.compare_joint26 import METHODS, compare


class Joint26ComparisonTest(unittest.TestCase):
    def make_runs(self, root, skipped=0):
        for method in METHODS:
            d = root / method
            (d / "val_scores").mkdir(parents=True)
            three, parax = method.startswith("THREE"), method.endswith("PARAX")
            config = dict(training_protocol="joint26", task_sizes=[26], max_tasks=1,
                          task_pathway_count=1, reporting_split="val", also_report_test=False,
                          adapter_mode="disabled", view_fusion="fixed_three_view" if three else "disabled",
                          parax_mode="image" if parax else "disabled", seed=0,
                          view_auxiliary_loss_weight=0.1 if three else 0, supervised_loss_scale=1 if three else 1.1,
                          epochs_per_task=1, train_batch_size=1)
            audit = dict(label_dimensions=26, test_loaded=False, frozen_visual_unchanged=True,
                         frozen_visual_sha256_before="clip", frozen_visual_sha256_after="clip",
                         train_instances=2, validation_instances=2, train_sample_ids_sha256="train",
                         validation_sample_ids_sha256="val", first_train_batch_ids_sha256="first",
                         initial_selector_prompt_classifier_sha256="init", peak_allocated_mib=1)
            row = dict(epoch=0, optimizer_steps=2-skipped, skipped_optimizer_steps=skipped,
                       current_loss=0.5, parax_grad_finite=1-skipped/2,
                       parax_grad_expert_a_norm=1, parax_grad_expert_b_norm=1, parax_grad_routers_0_norm=1)
            summary = dict(config=config, status="complete", test_metrics=None, completed_epochs=1,
                           completed_optimizer_updates=2-skipped, elapsed_seconds=1,
                           metrics=dict(final_mAP=40, final_cF1=20, final_oF1=20),
                           task_metrics=[dict(per_class_ap=[40]*26)])
            for name, data in (("seed_summary.json", summary), ("joint_protocol_audit.json", audit),
                               ("training_history.json", {"0": [row]}), ("view_diagnostics.json", {"0": {}})):
                (d / name).write_text(json.dumps(data))
            np.savez(d / "val_scores" / "task0.npz", sample_ids=np.array(["a", "b"]),
                     targets=np.ones((2, 26)), logits=np.zeros((2, 26)))

    def test_skips_require_explicit_acknowledgment_and_stay_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_runs(root, skipped=1)
            with self.assertRaisesRegex(ValueError, "skipped steps"):
                compare(root)
            result = compare(root, allow_skipped_updates=True)
            self.assertFalse(result["strict_zero_skip_audit_passed"])
            self.assertTrue(result["warnings"])

    def test_changed_backbone_or_labels_are_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_runs(root)
            self.assertTrue(compare(root)["strict_zero_skip_audit_passed"])
            d = root / "THREE_VIEW"
            audit = json.loads((d / "joint_protocol_audit.json").read_text())
            audit["frozen_visual_sha256_after"] = "changed"
            (d / "joint_protocol_audit.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "labels/backbone"):
                compare(root, allow_skipped_updates=True)
            audit["frozen_visual_sha256_after"] = "clip"
            (d / "joint_protocol_audit.json").write_text(json.dumps(audit))
            np.savez(d / "val_scores" / "task0.npz", sample_ids=np.array(["a", "b"]),
                     targets=np.zeros((2, 26)), logits=np.zeros((2, 26)))
            with self.assertRaisesRegex(ValueError, "samples/targets"):
                compare(root)
