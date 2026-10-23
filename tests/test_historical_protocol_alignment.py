from pathlib import Path
from types import SimpleNamespace
import unittest
from multi_lane.track_a.legacy_full_reference import historical_training_flags, historical_configuration_differences
from multi_lane.track_a.run_strong_full_asl import evaluation_job


class HistoricalProtocolTest(unittest.TestCase):
    def test_exact_paired_control_flags_are_preserved(self):
        flags=historical_training_flags(Path('data'),Path('clip'),Path('out'))
        self.assertIn('--paired-full-person',flags)
        for flag,value in [('--person-crop-margin','.15'),('--person-color-jitter-strength','.1'),
                           ('--person-color-jitter-probability','.2'),('--workers','2')]:
            self.assertEqual(flags[flags.index(flag)+1],value)
        for seed in [1,2]:
            flags=historical_training_flags(Path('data'),Path('clip'),Path('out'),seed=seed)
            self.assertEqual(flags[flags.index('--seed')+1],str(seed))

    def test_configuration_guard_accepts_only_checkpoint_retention_changes(self):
        archived=dict(paired_full_person=True,save_compact_checkpoints=False,workers=2)
        actual=dict(archived,save_compact_checkpoints=True)
        self.assertEqual(historical_configuration_differences(actual,archived),{})
        actual['paired_full_person']=False
        self.assertEqual(set(historical_configuration_differences(actual,archived)),{'paired_full_person'})

    def test_historical_evaluation_is_default_and_full_cohort_explicit(self):
        args=SimpleNamespace(data_root='data',clip_checkpoint='clip',face_manifest_root='face',smoke=False)
        self.assertNotIn('--full-cohort',evaluation_job(args,Path('out'),'T_ASL','test','test')['arguments'])
        args.evaluation_cohort='full'
        with self.assertRaises(ValueError):evaluation_job(args,Path('out'),'T_ASL','test','test')


if __name__=='__main__':unittest.main()
