import json
import unittest

import numpy as np

from scripts.emotic.d9_disagreement_fit import apply_task, fit_classifier, fit_task


def cache():
    rng = np.random.default_rng(1)
    reliable = np.array([True, False] * 4)
    z = np.zeros((len(reliable), 2, 3), dtype=np.float32)
    z[:, 0, 1] = [1, 2, 3, 4, 5, 6, 7, 8]
    z[:, 0, 2] = [1, 100, 2, 100, 5, 100, 9, 100]
    z[:, 1, 1:] = -1
    y = np.column_stack((reliable, ~reliable)).astype(np.uint8)
    return dict(branch_logits=z, targets=y, reliable=reliable, content=rng.normal(size=(8, 3, 8)),
                delta=rng.normal(size=(8, 2, 8)), relation=rng.normal(size=(8, 8)))


class DisagreementFitTest(unittest.TestCase):
    def test_train_cutoff_face_mask_and_no_evaluation_label_access(self):
        train = cache()
        model = json.loads(json.dumps(fit_task(train), allow_nan=False))
        self.assertEqual(model['classes'][0]['threshold'], 4.5)
        self.assertIsNone(model['classes'][1]['threshold'])
        eval_data = dict(train)
        del eval_data['targets']
        got = apply_task(eval_data, model)
        np.testing.assert_array_equal(got['receiver'][:, 0], [1, 1, 1, 1, 1, 1, 2, 1])
        np.testing.assert_array_equal(got['candidate'][:, 0], [False]*4 + [True]*4)
        self.assertFalse(got['candidate'][:, 1].any())
        for key in ('middle_probability', 'score_probability'):
            self.assertTrue(np.all(got[key][got['candidate']] > 0))
            self.assertTrue(np.all(got[key][~got['candidate']] == 0))
        eval_data['branch_logits'] = train['branch_logits'].copy()
        eval_data['branch_logits'][0,0,1] = 6
        self.assertTrue(apply_task(eval_data, model)['candidate'][0,0])

    def test_balanced_fit_intercept_and_regularization(self):
        x = np.array([[-1], [-1], [-1], [1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        fit = fit_classifier(x, y)
        b = fit['coef'][0]
        self.assertGreater(b, 0)
        self.assertAlmostEqual(fit['intercept'], 0., places=6)
        self.assertAlmostEqual(b, 4 / (1 + np.exp(b)), places=5)

    def test_missing_positive_training_label_never_accepts(self):
        d = cache()
        d['targets'][:] = 0
        model = fit_task(d)
        self.assertEqual(model['classes'][0]['gates'], {})
        r = apply_task(d, model)
        self.assertTrue(r['candidate'].any())
        self.assertFalse(np.any(r['middle_probability'] >= .5))
        self.assertFalse(np.any(r['score_probability'] >= .5))
