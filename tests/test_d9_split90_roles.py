import unittest
from collections import Counter
from types import SimpleNamespace

from scripts.emotic.d9_split90_roles import quota, split_roles


def records(rows):
    ids, reliable, targets = [], [], []
    for image, people in rows:
        for j, (valid, labels) in enumerate(people):
            ids.append(f'train:{image}#person={j}')
            reliable.append(valid)
            targets.append(labels)
    return SimpleNamespace(sample_ids=ids, face_reliable=reliable, targets=targets)


class SplitImageRolesTest(unittest.TestCase):
    def test_no_labels_read_for_metadata_and_hash(self):
        images = [(str(i), [(True, [0])] if i < 20 else [(True, [0]), (False, [1])]) for i in range(50)]
        data = records(images)
        del data.targets
        meta = split_roles(data, 'meta', 1)
        for strategy in ('meta', 'hash2'):
            roles = split_roles(data, strategy, 1)
            self.assertEqual(len(roles), 50)
            self.assertEqual(Counter(roles.values()), {'fusion': 5, 'branch': 45})
            self.assertEqual(roles, split_roles(records(images[::-1]), strategy, 1))
        self.assertEqual(sum(meta[f'train:{i}'] == 'fusion' for i in range(20)), 2)
        self.assertEqual(sum(meta[f'train:{i}'] == 'fusion' for i in range(20,50)), 3)

    def test_online_roles_permanent_after_first_visible_task(self):
        initial = [(str(i), [(bool(i % 3), [0]), (True, [1] if i % 2 else [])]) for i in range(35)]
        future = [(f'new{i}', [(True, [5] if i % 4 else [8])]) for i in range(15)]
        prefix = split_roles(records(initial), 'online', 1)
        changed = [(key, [(face, labels + [7, 8]) for face, labels in people]) for key, people in initial]
        final = split_roles(records(changed + future), 'online', 1)
        self.assertEqual(prefix, {k: final[k] for k in prefix})
        self.assertEqual(sum(v == 'fusion' for v in prefix.values()), quota(35))
        self.assertEqual(sum(v == 'fusion' for v in final.values()), quota(50))
        self.assertEqual(final, split_roles(records((changed + future)[::-1]), 'online', 1))

    def test_online_fills_remaining_unlabelled_and_never_splits_people(self):
        data = records([(str(i), [(True, [0, 1] if i < 14 else []), (False, [])]) for i in range(29)])
        roles = split_roles(data, 'online', 1)
        self.assertEqual(len(roles), 29)
        self.assertEqual(sum(roles[f'train:{i}'] == 'fusion' for i in range(14)), 1)
        self.assertEqual(sum(v == 'fusion' for v in roles.values()), 3)


if __name__ == '__main__':
    unittest.main()
