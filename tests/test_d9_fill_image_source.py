import json
from pathlib import Path
import tempfile
import unittest

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_fill_image_source import assert_train_roles, native_scores, source_mode


class ImageSourceTest(unittest.TestCase):
    def test_old_unequal_90_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = dict(seed=1, smoke=False, protect_branches=True, mode='sample_holdout',
                          image_split='global_10pct_hash_rank')
            (root/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                source_mode(root)
            config['equal_updates'] = {'total':13950}
            (root/'config.json').write_text(json.dumps(config))
            self.assertEqual(source_mode(root), 'sample_holdout')

    def test_exposure_roles_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'config.json').write_text(json.dumps(dict(seed=1,smoke=False,protect_branches=True,
                                                            mode='sample_full')))
            (root/'train_image_roles.json').write_text(json.dumps({'train:a':'both','train:b':'fusion'}))
            assert_train_roles(root,['train:a'])
            with self.assertRaises(ValueError):
                assert_train_roles(root,['train:b'])

    def test_native_cache_requires_matching_configuration_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base/'source'
            run = base/'run'
            for folder in (source/'cohorts', source/'compact_checkpoints', run/'native_train'):
                folder.mkdir(parents=True)
            (source/'config.json').write_text(json.dumps(dict(seed=1,smoke=False,protect_branches=True,
                                                              mode='sample_full')))
            (source/'cohorts/task0.json').write_text('{}')
            (source/'train_image_roles.json').write_text('{}')
            (source/'compact_checkpoints/task0.pth').write_bytes(b'checkpoint')
            archive = run/'native_train/task0.npz'
            archive.write_bytes(b'archive')
            protocol = {key+'_sha256':sha256_file(source/path) for key,path in
                        [('config','config.json'),('cohort','cohorts/task0.json'),
                         ('roles','train_image_roles.json'),('checkpoint','compact_checkpoints/task0.pth')]}
            archive.with_suffix('.json').write_text(json.dumps(dict(protocol=protocol,
                                                                      archive_sha256=sha256_file(archive))))
            self.assertEqual(native_scores(source,run/'features','train',0)[0],archive)
            archive.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                native_scores(source,run/'features','train',0)


if __name__ == '__main__':
    unittest.main()
