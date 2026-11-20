import unittest

from scripts.emotic.d9_split90_summary import COLUMNS, NAMES, above_both, report_markdown, with_absolute_reference


def scores(final, average):
    return {'final_mAP': final, 'average_mAP': average}


def table():
    return {key: {method: scores(i, i + 5) for method, _ in COLUMNS}
            for i, key in enumerate(NAMES)}


class Split90ReportTest(unittest.TestCase):
    def test_selection_uses_validation_both_controls_and_one_matching_branch(self):
        val, test = table(), table()
        for method, _ in COLUMNS:
            if method != 'fixed':
                val['online'][method]['average_mAP'] = val['hash2'][method]['average_mAP']
            test['meta'][method] = scores(-10, -11)
        report = report_markdown({'val': val, 'test': test})
        self.assertTrue(above_both(val, 'online', 'fixed'))
        self.assertIn('按当前任务情绪分层未通过候选条件', report)
        self.assertIn('按图中人数和人脸情况分层通过候选条件', report)
        self.assertIn('测试融合未在两项指标上同时高于两份随机名单', report)
        self.assertIn('| 按图中人数和人脸情况分层 | 第二份随机 90% | 可靠 Face | +1.0000 | +1.0000 | -11.0000 | -17.0000 |', report)
        self.assertEqual(report.count('| 全 TRAIN 参照 |'), 4)

    def test_full_train_comparison_must_match_all_historical_arms(self):
        history = table()['historical']
        reference = {'split': 'val', 'official_result_sha256': {'equal_updates': 'old'},
                     'branch_summary': {method: {group: {'values': {'equal_updates': history[method],
                                                                    'full': scores(42, 50)}}}
                                        for method, group in COLUMNS}}
        summary = {'split': 'val', 'table': {'historical': history}}
        self.assertEqual(with_absolute_reference(summary, {'old'}, reference)['full_train']['face'], scores(42, 50))
        with self.assertRaises(ValueError):
            with_absolute_reference(summary, {'different', 'old'}, reference)
        reference['branch_summary']['face']['reliable']['values']['equal_updates'] = scores(42, 50)
        with self.assertRaises(ValueError):
            with_absolute_reference(summary, {'old'}, reference)
