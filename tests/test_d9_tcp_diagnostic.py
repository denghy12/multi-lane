import unittest

import numpy as np

from scripts.emotic.d9_tcp_diagnostic import audit_rows, describe, summaries


class DiagnosticTest(unittest.TestCase):
    def test_tied_weights_have_no_spurious_order(self):
        w = np.full(8, .15)
        q = np.arange(8) / 8
        cell = describe(w, q, q - .1, np.ones(8))
        self.assertEqual(cell['overall']['n'], 8)
        self.assertIsNone(cell['high_minus_low']['tcp'])
        self.assertIsNone(cell['spearman']['tcp'])

    def test_transfer_is_paired_against_full_with_signed_bce(self):
        n, c = 16, 5
        r = np.arange(n) % 2 == 0
        y = np.tile((np.arange(n) % 4 < 2)[:, None], (1, c))
        z = np.zeros((n, c, 3))
        z[..., 1] = 1
        z[..., 2] = 2
        w = np.full(z.shape, 1/3.)
        rows = audit_rows(0, y, z, r, w)
        by_view = {(v['view'], v['reliable'], v['stratum']): v['overall'] for v in rows}
        self.assertNotIn(('face', False, 'positive'), by_view)
        for view, active, sign in [('face', True, 1), ('person', False, -1)]:
            base = .56 if active else .2
            change = .1 if view == 'face' else .05
            expected = np.logaddexp(0, -sign*base) - np.logaddexp(0, -sign*(base+change))
            measured = by_view[(view, active, 'positive' if sign == 1 else 'negative')]
            self.assertAlmostEqual(measured['bce_gain'], expected)
            self.assertAlmostEqual(measured['benefit_rate'], 1 if sign == 1 else 0)

    def test_quartiles_midrank_and_positive_cells(self):
        w = np.repeat([.1, .2, .3, .4], 2)
        q = np.linspace(.1, .8, 8)
        row = describe(w, q, q-.5, np.ones(8))
        self.assertEqual(row['high']['n'], 2)
        self.assertAlmostEqual(row['high_minus_low']['tcp'], .6)
        self.assertEqual(row['high_minus_low']['benefit_rate'], 1.)
        val = summaries([dict(view='face', stratum='all', **row)])
        self.assertEqual(val['face']['all']['spearman']['tcp']['positive_cells'], 1)
