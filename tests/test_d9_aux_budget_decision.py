import unittest

from scripts.emotic.d9_aux_budget_run import summarize


class DecisionRuleTest(unittest.TestCase):
    def test_macro_gain_from_one_task_does_not_qualify(self):
        reports = [{'macro': dict(fixed=30, group=30, shuffle=30, mean_person=30, person=x)}
                   for x in (34, 29, 29)]
        result = summarize(reports)
        self.assertTrue(result['criteria']['fixed'])
        self.assertFalse(result['eligible'])

    def test_fixed_is_a_floor_and_individual_advantage_is_required(self):
        reports = [{'macro': dict(fixed=30, group=29, shuffle=29, mean_person=29, person=30)}
                   for _ in range(3)]
        self.assertTrue(summarize(reports)['eligible'])
        reports[1]['macro']['mean_person'] = 32
        reports[2]['macro']['mean_person'] = 30
        self.assertFalse(summarize(reports)['eligible'])
