import json
import tempfile
import unittest
from pathlib import Path

import test_track_a_dual_view_fusion as fixtures
from multi_lane.track_a.compare_validation_ensembles import audit_validation_run, compare_ensembles


class ValidationEnsembleTest(unittest.TestCase):
    @staticmethod
    def make_runs(root):
        groups = [[], []]
        for group, view in enumerate(('full', 'person_crop')):
            for seed in range(3):
                run = root/f'{view}_{seed}'
                fixtures.DualViewScoreTest._write_validation_run(run, view, 0.01*seed)
                config = json.loads((run/'config.json').read_text())
                config.update(seed=seed, evaluation_score_purpose='validation_search',
                              git={'commit':f'commit{seed}', 'dirty':False})
                summary = json.loads((run/'seed_summary.json').read_text())
                summary.update(config=config, completed_epochs=240, completed_optimizer_updates=13950)
                (run/'config.json').write_text(json.dumps(config))
                (run/'seed_summary.json').write_text(json.dumps(summary))
                (run/'training_history.json').write_text(json.dumps({str(t):[{'skipped_optimizer_steps':0} for _ in range(30)] for t in range(8)}))
                groups[group].append(run)
        return groups

    def test_fixed_rule_matches_auxiliary_seeds_and_has_no_search(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            result = compare_ensembles(full, person)
            self.assertFalse(result['search_performed'])
            self.assertEqual(result['rule']['auxiliary_weight'], 0.2)
            self.assertEqual(result['auxiliary_seed_mapping'], [[0,1],[1,2],[2,0]])
            a, b = result['groups']['cyclic_full_full'], result['groups']['cyclic_full_person']
            self.assertEqual([(r['seed'],r['auxiliary_seed']) for r in a], [(r['seed'],r['auxiliary_seed']) for r in b])
            self.assertNotIn('winner', result)

    def test_rejects_test_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            config = json.loads((full[1]/'config.json').read_text())
            config['reporting_split']='test'
            (full[1]/'config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'test is forbidden'):
                compare_ensembles(full, person)

    def test_accepts_runner_val_scores_folder_name(self):
        with tempfile.TemporaryDirectory() as directory:
            full, _ = self.make_runs(Path(directory))
            (full[0]/'validation_scores').rename(full[0]/'val_scores')
            _, dumps, _ = audit_validation_run(full[0], 'full')
            self.assertEqual(len(dumps), 8)

    def test_accepts_audited_legacy_seed0_without_score_purpose_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            for run in (full[0], person[0]):
                config = json.loads((run/'config.json').read_text())
                del config['evaluation_score_purpose']
                summary = json.loads((run/'seed_summary.json').read_text())
                summary['config'] = config
                (run/'config.json').write_text(json.dumps(config))
                (run/'seed_summary.json').write_text(json.dumps(summary))
            self.assertEqual(compare_ensembles(full, person)['evaluation_split'], 'val')

    def test_rejects_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                compare_ensembles([full[0],full[0],full[2]],person)

    def test_rejects_changed_training_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            config = json.loads((full[1]/'config.json').read_text())
            config['learning_rate']=0.5
            summary=json.loads((full[1]/'seed_summary.json').read_text());summary['config']=config
            (full[1]/'config.json').write_text(json.dumps(config))
            (full[1]/'seed_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'configuration drift'):
                compare_ensembles(full,person)

    def test_rejects_corrupted_recorded_class_ap(self):
        with tempfile.TemporaryDirectory() as directory:
            full, person = self.make_runs(Path(directory))
            summary=json.loads((full[1]/'seed_summary.json').read_text())
            summary['task_metrics'][0]['per_class_ap'][0]-=1
            (full[1]/'seed_summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'per_class_ap'):
                compare_ensembles(full,person)


if __name__=='__main__':
    unittest.main()
