import unittest

from scripts.emotic.d9_quality_run import summarize


class PairedQualityCriteria(unittest.TestCase):
    def test_mean_gain_only_cannot_advance(self):
        reports = [dict(macro=dict(fixed=4, A=4, B=4, C=x, shuffle=4, mean_C=4)) for x in (8, 3, 3)]
        self.assertFalse(summarize(reports)['eligible'])

    def test_still_requires_the_fixed_reference(self):
        reports = [dict(macro=dict(fixed=5, A=4, B=4, C=4.5, shuffle=4, mean_C=4)) for _ in range(3)]
        self.assertFalse(summarize(reports)['eligible'])
        for report in reports:
            report['macro']['C'] = 5
        self.assertTrue(summarize(reports)['eligible'])
