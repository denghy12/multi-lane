import unittest
import torch

import test_protected_view_forward as fixtures
from multi_lane.track_a.fixed_model_mean_forward import FixedModelMeanFusion


class FixedModelMeanForwardTest(unittest.TestCase):
    pair=fixtures.ProtectedViewForwardTest.pair
    images=fixtures.ProtectedViewForwardTest.images

    def test_mean_is_after_source_view_fusion_before_one_common_head(self):
        a,b=self.pair(); wrapper=FixedModelMeanFusion(a,b); x=self.images()
        with torch.no_grad():
            features,sources=wrapper.sources(x)
            fa,wa=a.view_fusion_module(features['A'],a._lane_ids(True),x['face_reliable'])
            fb,wb=b.view_fusion_module(features['B'],b._lane_ids(True),x['face_reliable'])
            actual,weights=wrapper.compose_mean(x,features)
            expected=wrapper.combine(a.head(.5*fa+.5*fb))
        self.assertTrue(torch.equal(actual['fused'],expected))
        self.assertTrue(torch.equal(weights,wa) and torch.equal(weights,wb))
        # Both feature mean and logit mean must agree to FP32 rounding, including bias.
        self.assertTrue(torch.allclose(actual['fused'],.5*sources['A']['fused']+.5*sources['B']['fused'],atol=1e-6))

    def test_same_source_mean_replays_and_does_not_normalize_fused_feature(self):
        a,b=self.pair()
        wrapper=FixedModelMeanFusion(a,b); x=self.images()
        with torch.no_grad():
            features,sources=wrapper.sources(x)
            features['B']=features['A']
            mean,_=wrapper.compose_mean(x,features)
        self.assertTrue(torch.equal(mean['fused'],sources['A']['fused']))

    def test_unreliable_face_is_still_excluded_in_both_sources(self):
        a,b=self.pair(); wrapper=FixedModelMeanFusion(a,b); x=self.images()
        with torch.no_grad():
            features,_=wrapper.sources(x)
            before,_=wrapper.compose_mean(x,features)
            for source in features.values():
                source['face'][1].add_(1000)
            after,_=wrapper.compose_mean(x,features)
        self.assertTrue(torch.equal(before['fused'][1],after['fused'][1]))

    def test_mean_requires_B_and_runs_complete_source_dependencies(self):
        a,b=self.pair(); x=self.images(); wrapper=FixedModelMeanFusion(a,b)
        with torch.no_grad():
            mean=wrapper(x)
            self.assertEqual(set(mean),{'fused','full','person','face'})
            self.assertTrue(torch.equal(wrapper(x,'A')['fused'],a.seen_logits_with_views(x)[0]))
            single=FixedModelMeanFusion(a)
            self.assertTrue(torch.equal(single(x,'A')['fused'],wrapper(x,'A')['fused']))
            with self.assertRaises(ValueError):single(x)
