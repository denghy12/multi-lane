import copy
import unittest
from types import SimpleNamespace

from scripts.emotic.d9_equal_update_budget import check_epoch_budget, epoch_budget


class EqualUpdateBudgetTest(unittest.TestCase):
    def test_historical_partial_batch_and_rejected_skip(self):
        row = dict(task=3, epoch=1, examples=130, updates=3, skipped=0)
        history = dict(coverage=dict(count=130), epochs=[row])
        self.assertEqual(epoch_budget(history, 3, 130, 1, 64),
                         dict(examples=130, updates=3, last_batch_size=2))
        for key, wrong in [('epoch', 0), ('examples', 129), ('updates', 2), ('skipped', 1)]:
            with self.subTest(key=key):
                broken = copy.deepcopy(history)
                broken['epochs'][0][key] = wrong
                with self.assertRaises(ValueError):
                    epoch_budget(broken, 3, 130, 1, 64)

    def test_checks_actual_batch_sizes_and_cosine_clock(self):
        config = dict(smoke=False, epochs=2, batch_size=64,
                      equal_updates=dict(tasks={'3': dict(examples=130)}))
        opt = SimpleNamespace(param_groups=[{'lr': .005}, {'lr': .0002}])
        sched = SimpleNamespace(last_epoch=1, base_lrs=[.010, .0004])
        row = dict(updates=3, examples=130, skipped=0,
                   sampling=dict(unique_people=114, examples=130, batch_sizes=[64, 64, 2],
                                 duplicate_within_batches=0))
        check_epoch_budget(3, 2, config, opt, sched, row)
        self.assertEqual(row['branch_lrs'], [.005, .0002])
        row['sampling']['batch_sizes'] = [63, 65, 2]
        with self.assertRaises(RuntimeError):
            check_epoch_budget(3, 2, config, opt, sched, row)
        row['sampling']['batch_sizes'] = [64, 64, 2]
        opt.param_groups[0]['lr'] = .010
        with self.assertRaises(RuntimeError):
            check_epoch_budget(3, 2, config, opt, sched, row)
