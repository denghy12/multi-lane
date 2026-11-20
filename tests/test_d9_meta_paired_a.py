"""The archived metadata-split fit is opt-in; the original online guard stays active."""
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_equal_updates_paired import check_fit_point, fit, validate_source


def recipe(smoke=False):
    return dict(mode='sample_holdout', seed=1, smoke=smoke, tasks=2 if smoke else 8,
                batch_size=4 if smoke else 64, epochs=1 if smoke else 30, protect_branches=True,
                image_split='global_10pct_meta', split90='meta',
                equal_updates={'mode': 'cycle_90pct_to_full_epoch_size', 'roles_sha256': 'from-training'})


class MetaPairedReplayTest(unittest.TestCase):
    def test_replay_only_accepts_meta_trajectory(self):
        for smoke in (False, True):
            cfg = recipe(smoke)
            validate_source(cfg, True)
            with self.assertRaises(ValueError):
                validate_source(cfg, False)
            random_cfg = dict(cfg, image_split='global_10pct_hash_rank')
            random_cfg.pop('split90')
            validate_source(random_cfg, False)
            with self.assertRaises(ValueError):
                validate_source(random_cfg, True)
            for edit in ({'split90': 'online'}, {'epochs': 35}, {'protect_branches': False}):
                with self.subTest(edit=edit), self.assertRaises(ValueError):
                    validate_source(dict(cfg, **edit), True)

    def test_next_task_guard_still_applies_to_online_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'cohorts').mkdir()
            (root / 'cohorts/task1.json').touch()
            with self.assertRaisesRegex(ValueError, 'before the next task starts'):
                check_fit_point(root, 0, False)
            check_fit_point(root, 0, True)

    def test_replay_checks_training_role_digest_and_previous_center(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'cohorts').mkdir()
            (root / 'compact_checkpoints').mkdir()
            (root / 'config.json').write_text('{}')
            roles = root / 'train_image_roles.json'
            roles.write_text('{}')
            for task in (0, 1):
                (root / 'cohorts' / f'task{task}.json').write_text('{}')
                (root / 'compact_checkpoints' / f'task{task}.pth').touch()
            cfg = recipe()
            with self.assertRaisesRegex(ValueError, 'TRAIN role manifest'):
                fit(root, root, cfg, 0, 0, True)
            cfg['equal_updates']['roles_sha256'] = sha256_file(roles)
            with self.assertRaises(FileNotFoundError):
                fit(root, root, cfg, 1, 0, True)


if __name__ == '__main__':
    unittest.main()
