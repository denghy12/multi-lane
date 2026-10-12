import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_lane.track_a.compare_whole_parax_freeze import audit_control
from multi_lane.track_a.evaluate_whole_parax_eight_task import evaluate, filter_seen_people, seen_payload, summarize_curve
from multi_lane.track_a.runner import TASK_SIZES


class WholeParaXEightTaskTest(unittest.TestCase):
    def fixture(self, root):
        source, reference = root/'frozen', root/'reference'
        source.mkdir(); reference.mkdir()
        # Only provenance fields consumed by the audit are needed in this fixture.
        c = dict(seed=0, max_tasks=8, task_sizes=list(TASK_SIZES), parax_mode='image',
                 parax_freeze_all_after_task0=True, image_stream_transfer_audit=True,
                 reporting_split='val', also_report_test=False, git={'dirty': False},
                 train_batch_size=64, parax_optimizer_initial_learning_rate=.0004)
        for key in ('class_order', 'eval_batch_size', 'learning_rate', 'scheduler_mode',
                    'input_normalization', 'train_crop_scale', 'selector_mode', 'prompt_mode',
                    'view_fusion', 'view_classifier_mode', 'view_auxiliary_loss_weight',
                    'supervised_loss_scale', 'training_loss_mode', 'parameter_group_loss_routing',
                    'adapter_mode', 'parax_layer_indices', 'parax_rank', 'parax_num_experts',
                    'parax_router_hidden', 'parax_initialization', 'parax_residual_scale',
                    'parax_output_scale_mode', 'parax_freeze_center_after_task0',
                    'parax_task_local_gate', 'parax_level_conditioned', 'amp_initial_scale',
                    'amp_growth_interval', 'tf32'):
            c[key] = 'same'
        tasks = {}
        for t in range(8):
            tasks[str(t)] = dict(train_instances=64, test_loaded=False)
            for key in ('sampler_seed', 'train_sample_ids_sha256', 'first_train_batch_ids_sha256',
                        'validation_sample_ids_sha256', 'reporting_sample_ids_sha256', 'frozen_visual_sha256_before'):
                tasks[str(t)][key] = str(t)
        a = dict(frozen_visual_unchanged=True, frozen_visual_sha256_before='clip',
                 frozen_visual_sha256_after='clip', initial_selector_prompt_classifier_sha256='initial', tasks=tasks)
        h = {str(t): [dict(optimizer_steps=1, skipped_optimizer_steps=0, amp_loss_scale_end=1024,
                           current_loss=.5, parax_grad_finite=1, parax_grad_total_norm=1 if t == 0 else 0,
                           adapter_learning_rate=.0004) for _ in range(30)] for t in range(8)}
        d = dict(tasks={str(t): dict(old_lane_parameters_unchanged=True, whole_parax_freeze_verified=t>0,
                                    parax_trainable_parameters=1 if t == 0 else 0, shared_parax_changed=t==0,
                                    shared_parax_state_hash_after='frozen') for t in range(8)})
        for p in (source, reference):
            (p/'config.json').write_text(json.dumps(c))
            (p/'paired_protocol_audit.json').write_text(json.dumps(a))
        for name, value in [('seed_summary.json', dict(status='complete', config=c, test_metrics=None, completed_optimizer_updates=240)),
                            ('training_history.json', h), ('image_stream_transfer_audit.json', d)]:
            (source/name).write_text(json.dumps(value))
        for p in (source, reference):
            (p/'val_scores').mkdir()
            np.savez(p/'val_scores/task0.npz', sample_ids=np.array(['a']), targets=np.zeros((1,5)),
                     class_indices=np.arange(5), logits=np.zeros((1,5)))
        return source, reference, d

    def test_eight_task_audit_checks_late_task_and_rejects_three_task_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, reference, d = self.fixture(Path(tmp))
            _, result = audit_control(source, reference, task_count=8)
            self.assertEqual(result['optimization_updates'], 240)
            with self.assertRaises(ValueError):
                audit_control(source, reference)  # Default pilot still requires max_tasks=3.
            d['tasks']['7']['shared_parax_state_hash_after'] = 'changed'
            (source/'image_stream_transfer_audit.json').write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'Freeze failed'):
                audit_control(source, reference, task_count=8)

    def test_payload_excludes_future_labels_but_retains_negative_people(self):
        targets = np.zeros((3,26)); targets[0,0] = 1; targets[1,20] = 1
        full = dict(sample_ids=np.array(['a','b','c']), full_targets=targets,
                    face_reliable=np.array([True,False,True]), logits=np.zeros((3,5)),
                    probabilities=np.zeros((3,5)), batch_lengths=np.array([3]))
        data = seen_payload(full, 0)
        self.assertEqual(data['targets'].shape, (3,5))
        np.testing.assert_array_equal(filter_seen_people(data)['sample_ids'], ['a'])
        self.assertNotIn('full_targets', data)
        self.assertNotIn('batch_lengths', data)

    def test_fixed_cohort_forgetting_does_not_treat_harder_new_classes_as_old_forgetting(self):
        tasks = {}
        for t in range(8):
            ap = [80.]*5+[10.]*(sum(TASK_SIZES[:t+1])-5)
            tasks[str(t)] = dict(metrics=dict(fused=dict(per_class_ap=ap, seen_classes=len(ap), mAP=float(np.mean(ap)))))
        summary = summarize_curve(tasks)
        self.assertLess(summary['final_mAP'], 80)
        self.assertEqual(summary['forgetting'], 0)

    def test_test_preflight_is_rejected_before_loading_any_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'complete locked run'):
                evaluate(p,p,None,p/'output',p,p,p,split='test',preflight=True)


if __name__ == '__main__':
    unittest.main()
