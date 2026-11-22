import unittest

import numpy as np
import torch

from scripts.emotic.d9_image_gate_readout import Stage
from scripts.emotic.d9_image_shrink_fit import GRID, assign_folds, choose, subset_images, task_curve


class ImageShrinkFitTest(unittest.TestCase):
    def test_fold_is_a_function_of_image_across_tasks(self):
        names = np.asarray(['train:a','train:b','train:c'])
        self.assertEqual(assign_folds(names)[0], assign_folds(names[[2,0]])[1])
        self.assertTrue(((assign_folds(names) >= 0) & (assign_folds(names) < 3)).all())

    def test_subset_preserves_multi_person_image_identity(self):
        rows = torch.tensor([1,0,2,1,0])
        vec = torch.arange(3.).reshape(3,1)
        x = Stage(np.asarray(list('xyz')),vec,vec, np.array([3,2,1]),np.array([2,0,1]),
                  torch.tensor([False,False,True]),rows,torch.tensor([2,2,1]),torch.zeros(5).bool(),
                  torch.ones(5,1,3),torch.arange(5.).reshape(5,1),torch.empty(0),'','','')
        sub = subset_images(x,np.array([True,False,True]))
        self.assertEqual(sub.image_ids.tolist(), ['x','z'])
        self.assertEqual(sub.inverse.tolist(),[0,1,0])
        self.assertEqual(sub.targets[:,0].tolist(),[1.,2.,4.])
        self.assertEqual(sub.observed.tolist(),[2,1])

    def test_equal_class_and_stage_weight_tie_break(self):
        fold = np.tile([[1.,5.,np.nan]],(len(GRID),1))
        second = np.tile([[np.nan,1.,np.nan]],(len(GRID),1))
        curve = task_curve([fold,second])
        np.testing.assert_array_equal(curve, np.full(len(GRID),2.))
        self.assertEqual(choose([curve,curve])[0],0.)
        self.assertIsNone(task_curve([np.full_like(fold,np.nan)]))
        late = curve.copy()
        late[3] += 1
        self.assertEqual(choose([curve,late,None])[0], GRID[3])
