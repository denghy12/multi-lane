import unittest

import torch

from scripts.emotic.d9_image_shrink_eval import mix, summarize


class ImageShrinkEvalTest(unittest.TestCase):
    def test_endpoints_replay_and_only_supported_views_receive_mass(self):
        a=torch.tensor([[.8,.2,0],[.7,.2,.1]])
        w=torch.tensor([[.3,.7,0],[.1,.35,.55]])
        self.assertIs(mix(a,w,0),a)
        self.assertIs(mix(a,w,1),w)
        v=mix(a,w,.25)
        torch.testing.assert_close(v, torch.tensor([[.675,.325,0],[.55,.2375,.2125]]))
        for amplitude in (float('nan'), -.1, 1.01):
            with self.assertRaises(ValueError):
                mix(a,w,amplitude)

    def test_partial_subgroups_do_not_get_eight_stage_label(self):
        stage_0 = {'metrics':{'m':{'all':{'mAP':6.},'rare':{'mAP':9.}}}}
        stage_1 = {'metrics':{'m':{'all':{'mAP':4.}}}}
        self.assertEqual(summarize([stage_0,stage_1]),
                         {'m':{'all':{'final_mAP':4.,'average_mAP':5.}}})


if __name__=='__main__':
    unittest.main()
