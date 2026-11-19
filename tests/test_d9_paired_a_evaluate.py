import unittest

from scripts.emotic.d9_paired_a_evaluate import GROUPS, diff_by_stage
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES


class EightStageAttributionTest(unittest.TestCase):
    def test_old_and_new_heads_use_their_own_class_counts(self):
        size0, size1 = TASK_SIZES[:2]
        metrics = {}
        for name, old, new in [('natural', 10., 20.), ('paired', 12., 17.), ('fixed', 6., 15.)]:
            ap = [old] * size0 + [new] * size1
            metrics[name] = {g:dict(per_class_ap=ap, mAP=sum(ap)/len(ap)) for g in GROUPS}
        d = diff_by_stage(1, metrics)
        for g in GROUPS:
            self.assertEqual(d['paired_minus_natural'][g]['new_classes'], -3.)
            self.assertEqual(d['paired_minus_natural'][g]['old_classes'], 2.)
            self.assertEqual(d['paired_minus_fixed'][g]['per_class'][CLASS_ORDER[size0]], 2.)
