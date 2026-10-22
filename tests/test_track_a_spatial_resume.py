import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from multi_lane.track_a.spatial_resume import (
    checkpoint_source_git, prepare_boundary, restore_boundary_model, validate_boundary,
)
from multi_lane.track_a.runner import compact_model_state_dict
import test_track_a_spatial_summary as spatial_fixtures


class SpatialResumeTest(unittest.TestCase):
    def fixture(self):
        config = dict(git=dict(commit='old', dirty=False), adapter_mode='disabled',
            parax_mode='disabled', joint_summary_mode='disabled', cross_view_mode='disabled',
            selector_conditioning='disabled', view_fusion='fixed_three_view',
            view_classifier_mode='shared_post_fusion', selector_mode='shared', prompt_mode='shared',
            workers=0, eval_workers=0, reporting_split='val', also_report_test=False,
            skip_validation_eval=True, moe_paired_audit=True, save_pre_evaluation_recovery=True,
            spatial_summary_mode='target', max_tasks=2, optimizer_updates_per_task=2, epochs_per_task=None)
        payload = dict(schema_version=1, kind='task_boundary_before_reporting',
            next_action='report_completed_task_then_activate_next', task_id=0,
            source_git=config['git'], config=config, sampler_generator=torch.Generator().get_state(),
            rng=dict(torch_cuda=[torch.tensor([1])]),
            training_state=dict(optimizer={}, scheduler={}, scaler={}, optimizer_parameter_names=[]),
            provenance=dict(training_history={'0':[dict(optimizer_steps=2)]}, completed_reporting=[]))
        current = copy.deepcopy(config); current['git'] = dict(commit='new', dirty=False)
        return config, payload, current

    def test_reject_changed_learning_setup_incomplete_rng_and_wrong_profile(self):
        _, payload, current = self.fixture()
        validate_boundary(payload, current)
        changed = copy.deepcopy(current); changed['max_tasks'] = 8
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            validate_boundary(payload, changed)
        changed = copy.deepcopy(payload); changed['rng']['torch_cuda'] = []
        with self.assertRaisesRegex(ValueError, 'RNG'):
            validate_boundary(changed, current)
        changed = copy.deepcopy(payload); changed['config']['workers'] = 2
        changed_current = copy.deepcopy(current); changed_current['workers'] = 2
        with self.assertRaisesRegex(ValueError, 'profile'):
            validate_boundary(changed, changed_current)

    def test_inherited_artifacts_are_copies_and_original_git_is_checked_per_task(self):
        old, payload, current = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'output'
            output.mkdir(); (source/'recovery_checkpoints').mkdir(parents=True)
            (source/'calibration_split_provenance').mkdir()
            (source/'config.json').write_text(json.dumps(old))
            (source/'calibration_split_provenance/task0.json').write_text('{}')
            path = source/'recovery_checkpoints/task0_before_reporting.pth';torch.save(payload,path)
            before = path.read_bytes()
            actual = prepare_boundary(path,current,output)
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(before,(output/'recovery_checkpoints'/path.name).read_bytes())
            self.assertEqual(actual['inherited_calibration_rows'],[])
            self.assertEqual(current['resumed_from']['inherited_updates'],2)
            self.assertEqual(checkpoint_source_git(current,0),current['git'])
        current['resumed_from']['boundary_task']=1
        self.assertEqual(checkpoint_source_git(current,0),old['git'])
        self.assertEqual(checkpoint_source_git(current,1),current['git'])
        self.assertEqual(checkpoint_source_git(old,0),old['git'])

    def test_restore_keeps_trained_boundary_and_future_activation_copies_it(self):
        model = spatial_fixtures.SpatialSummaryTest().model(); model.activate_task(0)
        with torch.no_grad():
            model.selectors[0].add_(.3)
            model.spatial_summary_bank.tasks[0].add_(.2)
        state = copy.deepcopy(compact_model_state_dict(model))
        payload = dict(model=state, task_id=0, requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
                       model_training=True)
        restored = spatial_fixtures.SpatialSummaryTest().model()
        restore_boundary_model(restored,payload)
        self.assertTrue(all(torch.equal(v,compact_model_state_dict(restored)[n]) for n,v in state.items()))
        restored.activate_task(1)
        self.assertTrue(torch.equal(restored.selectors[1],state['selectors'][0]))
        self.assertTrue(torch.equal(restored.spatial_summary_bank.tasks[0],state['spatial_summary_bank.tasks.0']))
        self.assertFalse(restored.spatial_summary_bank.tasks[0].requires_grad)

    def test_process_guard_refuses_unrelated_process(self):
        import os
        from multi_lane.track_a.pause_spatial_boundary import identity
        if not Path('/proc').exists():self.skipTest('Linux process identity guard')
        with self.assertRaisesRegex(RuntimeError,'Unexpected process'):
            identity(os.getpid(),'not-the-running-experiment',Path.cwd())

    def test_continuation_keeps_training_arguments_and_source_command(self):
        from multi_lane.track_a.continue_spatial_gpu0 import command_arguments
        command = ['python','-c','runpy','--training-protocol','incremental',
                   '--epochs','30','--train-batch-size','64','--output-root','old/train']
        original = list(command)
        actual = command_arguments({'command':command},Path('new/train'),Path('saved.pth'))
        self.assertEqual(command,original)
        self.assertEqual(actual,['--training-protocol','incremental','--epochs','30',
                                 '--train-batch-size','64','--output-root','new/train',
                                 '--resume-spatial-boundary','saved.pth'])


if __name__ == '__main__':
    unittest.main()
