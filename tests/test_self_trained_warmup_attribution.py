import copy
import unittest
import torch

import test_protected_view_forward as fixtures
from multi_lane.track_a.self_trained_warmup_attribution import compare_boundaries
from multi_lane.track_a.historical_b32_backbone_attribution import method_state


class WarmupAttributionTest(unittest.TestCase):
    def test_current_representation_can_change_but_head_and_old_task_cannot(self):
        m,_=fixtures.ProtectedViewForwardTest.pair(self)
        w=copy.deepcopy(method_state(m));f=copy.deepcopy(w)
        f['selectors'][1].add_(.1)
        pair=compare_boundaries(w,f,1,m._task_sizes)
        self.assertTrue(pair['own_head_exact']);self.assertFalse(pair['all_parameters_exact'])
        f['selectors'][0].add_(.1)
        with self.assertRaises(AssertionError):compare_boundaries(w,f,1,m._task_sizes)

    def test_baseline_is_identity_and_external_head_swap_is_rejected(self):
        m,_=fixtures.ProtectedViewForwardTest.pair(self)
        w=copy.deepcopy(method_state(m));f=copy.deepcopy(w)
        self.assertTrue(compare_boundaries(w,f,0,m._task_sizes,True)['all_parameters_exact'])
        f['head.weight'].add_(.01)
        with self.assertRaises(AssertionError):compare_boundaries(w,f,0,m._task_sizes)
