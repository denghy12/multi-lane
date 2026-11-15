import copy
import sys
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.r1_crossvit_experiment import restore_compact_stage
from multi_lane.track_a.r1_frozen_cross_view import FrozenR1Receiver
from multi_lane.track_a.runner import (
    CLASS_ORDER, TASK_PROTOCOLS, TASK_SIZES, compact_model_state_dict, compute_metrics,
    config_task_sizes, dataset_view, parse_args, seen_indices, summarize_tasks,
    task_indices, train_task,
)
from test_track_a_reproduction import FakeVisual


class D9BenchmarkTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cli_defaults_match_benchmark_stages(self):
        base = ['runner', '--seed', '1', '--data-root', 'data', '--clip-checkpoint', 'clip',
                '--output-root', 'out']
        for name, count in [('b5c3', 8), ('b10c4', 5), ('b4c2', 12), ('joint26', 1)]:
            with self.subTest(protocol=name):
                extra = [] if name == 'b5c3' else ['--task-protocol', name]
                with patch.object(sys, 'argv', base + extra):
                    args = parse_args()
                self.assertEqual((args.task_protocol, args.max_tasks), (name, count))
                with patch.object(sys, 'argv', base + extra + ['--max-tasks', '1']):
                    self.assertEqual(parse_args().max_tasks, 1)

    def test_stage_classes_and_population(self):
        class Source:
            targets = [[0], [3, 10], [24], [25]]

        for name, first, last in [('b5c3', 5, 3), ('b10c4', 10, 4), ('b4c2', 4, 2),
                                  ('joint26',26,26)]:
            sizes = TASK_PROTOCOLS[name]
            ids = [task_indices(stage, sizes) for stage in range(len(sizes))]
            self.assertEqual(ids[0], tuple(range(first)))
            self.assertEqual(ids[-1], tuple(range(26-last, 26)))
            self.assertEqual([i for stage in ids for i in stage], list(range(26)))
            for stage in range(len(sizes)):
                seen = seen_indices(stage, sizes)
                view = dataset_view(Source(), seen)
                self.assertEqual(view.class_indices, tuple(range(sum(sizes[:stage+1]))))
                self.assertEqual(view.indices, tuple(i for i, y in enumerate(Source.targets)
                                                    if set(y) & set(seen)))

    def test_checkpoint_schedule_validation(self):
        self.assertEqual(config_task_sizes({}), TASK_SIZES)
        for name, sizes in TASK_PROTOCOLS.items():
            config = dict(task_sizes=list(sizes), class_order=list(CLASS_ORDER),
                          protocol_id=f'emotic_{name}_v0.1', max_tasks=len(sizes))
            self.assertEqual(config_task_sizes(config), sizes)
            with self.assertRaises(ValueError):
                config_task_sizes(dict(config, max_tasks=len(sizes)+1))
            with self.assertRaises(ValueError):
                config_task_sizes(dict(config, class_order=list(reversed(CLASS_ORDER))))
            with self.assertRaises(ValueError):
                config_task_sizes(dict(config, protocol_id='wrong'))

    def test_final_stage_real_train_restore_and_readout(self):
        for name in ('b10c4', 'b4c2', 'joint26'):
            with self.subTest(protocol=name):
                torch.manual_seed(18)
                sizes = TASK_PROTOCOLS[name]
                visual = FakeVisual()
                visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
                model = FrozenR1Receiver(
                    visual, sizes, num_selectors=3, num_prompts=2, num_prompt_layers=1,
                    adapter_mode='image_token', adapter_bottleneck_dim=4, adapter_layer_indices=(0,),
                    r1_view='person', r1_layer=0, r1_dim=8, r1_heads=2, frozen_cross_mode='patch')
                for stage in range(len(sizes)):
                    model.activate_task(stage)
                x = torch.randn(4, 3, 4, 4)
                inputs = dict(full=x, evidence_full=x*.8, evidence_person=x*.7, evidence_face=x*.6,
                              evidence_face_reliable=torch.tensor([True, False, True, False]))
                target = torch.tensor([[1.], [0.], [0.], [1.]]).expand(-1, sizes[-1])
                batch = [(inputs, target)]
                history = train_task(model, batch, batch, torch.device('cpu'), stage, 1, .001, 0.,
                                     1., False, adapter_learning_rate=.0004, loss_routing='adapter_asl')
                self.assertEqual(history[-1]['optimizer_steps'], 1)
                self.assertEqual(history[-1]['skipped_optimizer_steps'], 0)
                model.eval()
                expected = model.seen_logits(inputs).detach()
                self.assertEqual(expected.shape, (4, 26))
                checkpoint = dict(task_id=stage, task_sizes=list(sizes),
                                  model=compact_model_state_dict(model))
                restored = copy.deepcopy(model)
                restore_compact_stage(restored, checkpoint, stage, sizes)
                torch.testing.assert_close(restored.seen_logits(inputs), expected, rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    restore_compact_stage(restored, checkpoint, stage+1, sizes)
                with self.assertRaises(ValueError):
                    restore_compact_stage(restored, checkpoint, stage, TASK_SIZES)
                with self.assertRaises(ValueError):
                    restore_compact_stage(restored, {'task_id': stage}, stage, sizes)

    def test_forgetting_counts_and_joint_avg(self):
        for sizes in TASK_PROTOCOLS.values():
            rows = []
            for task, _ in enumerate(sizes):
                width = sum(sizes[:task+1])
                scores = torch.tensor([[.1], [.9]]).expand(-1, width)
                rows.append(compute_metrics(task, scores, scores.round(), .5))
            result = summarize_tasks(rows, sizes)
            self.assertEqual(len(result['per_class_forgetting']), 26-sizes[-1])
            self.assertAlmostEqual(result['final_mAP'], result['average_mAP'], places=10)
            if len(sizes) == 1:
                self.assertEqual(result['final_mAP'], result['average_mAP'])
                self.assertEqual(result['forgetting'], 0.)


if __name__ == '__main__':
    unittest.main()
