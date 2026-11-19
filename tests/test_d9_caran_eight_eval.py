import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_caran_eight_eval import combine, group_means, shuffle_scores, validation_seal
from scripts.emotic.d9_direction_permutation import batch_map, permutation_sources


class CaranReadoutEvaluationTest(unittest.TestCase):
    def test_weight_matrix_from_one_donor_per_head(self):
        rng = np.random.default_rng(9)
        sizes = [2, 3]
        n = 17
        ids = np.asarray([f'image{i}' for i in range(n)])
        valid = np.arange(n) % 2 == 0
        w = rng.uniform(size=(n,sum(sizes),3)).astype('float32')
        w[~valid,:,2] = 0
        w /= w.sum(-1,keepdims=True)
        z = rng.normal(size=w.shape).astype('float32')
        y = rng.integers(2, size=z.shape[:2])
        actual = batch_map(combine(w,z)[None], y)[0]
        w2 = w[:,::-1,:].copy()
        actual2 = batch_map(combine(w2,z)[None], y)[0]
        seeds = [193, 201, 218]
        output = shuffle_scores(ids,valid,sizes,{'D':w,'E':w2},z,y,{'D':actual,'E':actual2},seeds)
        for m, matrix, measured in [('D',w,actual),('E',w2,actual2)]:
            expected = []
            for seed in seeds:
                donors = permutation_sources(ids,valid,len(sizes),seed)
                swapped = np.empty_like(matrix)
                for person in range(n):
                    swapped[person,:2] = matrix[donors[person,0],:2]
                    swapped[person,2:] = matrix[donors[person,1],2:]
                expected.append(measured - batch_map(combine(swapped,z)[None],y)[0])
            np.testing.assert_allclose(output[m]['differences'],expected,atol=0,rtol=0)

    def test_group_mean_preserves_class_and_availability(self):
        valid = torch.tensor([False,False,True,True])
        w = torch.tensor([[[1.,0.,0.],[.5,.5,0]],[[.6,.4,0],[.3,.7,0]],
                          [[.3,.2,.5],[0,0,1]],[[.5,.4,.1],[1,0,0]]])
        mu = group_means(w,valid)
        torch.testing.assert_close(mu[0],torch.tensor([[.8,.2,0],[.4,.6,0]]))
        torch.testing.assert_close(mu[1],torch.tensor([[.4,.3,.3],[.5,0,.5]]))
        torch.testing.assert_close(mu[valid.long()].sum(-1),torch.ones(4,2))

    def test_changed_fits_cannot_reuse_validation_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            val=out/'val_results.json'
            val.write_text('{}')
            lock=dict(caran_fit_lock_sha256='first',val_results_sha256=sha256_file(val))
            (out/'validation_seal.json').write_text(json.dumps(lock))
            self.assertEqual(validation_seal(out,'first'),lock)
            with self.assertRaises(ValueError):
                validation_seal(out,'different')
            val.write_text('modified')
            with self.assertRaises(ValueError):
                validation_seal(out,'first')
