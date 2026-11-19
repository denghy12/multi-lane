import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_tcp_probe import identify_source


class BranchSourceModeTests(unittest.TestCase):
    def test_feature_archives_from_the_two_branch_protocols_cannot_be_interchanged(self):
        with TemporaryDirectory() as temp:
            prior = Path(temp)
            path = prior / 'features/task6/val.npz'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'fixture')
            protocol = dict(task=6, split='val', smoke=False, checkpoint_sha256='cp',
                            source_config_sha256='cfg')
            meta = dict(protocol=protocol, features_sha256=sha256_file(path),
                        exact_eval_replay=True, visibility_mask_applied=True)
            path.with_suffix('.json').write_text(json.dumps(meta))
            self.assertEqual(identify_source(prior, 6, 'val')[0], path)
            with self.assertRaises(ValueError):
                identify_source(prior, 6, 'val', 'sample_full')
            protocol['source_mode'] = 'sample_full'
            path.with_suffix('.json').write_text(json.dumps(meta))
            self.assertEqual(identify_source(prior, 6, 'val', 'sample_full')[0], path)
            with self.assertRaises(ValueError):
                identify_source(prior, 6, 'val')
