import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/emotic'))
from analyze_person_grounded_breakpoints import adjacent_pairs,auc,cluster_interval,strata


class BreakpointAnalysisTests(unittest.TestCase):
    def test_auc_ties_are_half_correct(self):
        self.assertEqual(auc(np.array([0.,1.,2.,3.]),np.array([0,1,0,1])),75.)
        self.assertEqual(auc(np.ones(4),np.array([0,1,0,1])),50.)

    def test_adjacent_negative_pairs_track_repairs_and_damage(self):
        base=np.array([.2,.1,.3]);projection=np.array([.3,0.,0.]);y=np.array([1,0,0])
        m=adjacent_pairs(base,base+projection,projection,y)
        self.assertEqual(m['pairs'],2);self.assertEqual(m['repaired_fraction'],.5)
        self.assertEqual(m['damaged_fraction'],0.);self.assertEqual(m['favorable_fraction'],1.)

    def test_geometry_stratum_cannot_claim_binding_from_one_large_box(self):
        z=dict(eligible=np.ones(2,bool),own_valid=np.ones(2,bool),other_valid=np.ones(2,bool),
            high_visibility=np.ones(2,bool),own_area=np.array([.5,.2]),other_area=np.array([.1,.2]),
            bbox_iou=np.array([0.,0.]),own_exclusive_area=np.array([.5,.2]),
            other_exclusive_area=np.array([.1,.2]),different_seen_labels=np.ones((2,5),bool))
        self.assertEqual(strata(z,0)['high_exclusive_size_matched'].tolist(),[False,True])

    def test_cluster_resampling_keeps_people_from_same_image_together(self):
        lo,hi=cluster_interval(np.array([1.,1.,1.,-1.]),
            np.array(['a#person=0','a#person=1','a#person=2','b#person=0']),500)
        self.assertEqual((lo,hi),(-1.,1.))


if __name__=='__main__':unittest.main()
