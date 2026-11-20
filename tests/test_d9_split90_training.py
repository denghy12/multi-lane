import hashlib
import json
import unittest

from scripts.emotic.d9_equal_update_budget import validate_reassigned_roles


class AlternativeSplitTrainingTest(unittest.TestCase):
    def setUp(self):
        self.historical = {str(i): ('fusion' if i == 1 else 'branch') for i in range(10)}
        self.candidate = {str(i): ('fusion' if i == 9 else 'branch') for i in range(10)}

    def test_membership_can_change_but_hash_locks_the_selected_split(self):
        self.assertNotEqual(self.candidate, self.historical)
        expected = hashlib.sha256((json.dumps(self.candidate, indent=2) + '\n').encode()).hexdigest()
        self.assertEqual(validate_reassigned_roles(self.historical, self.candidate), expected)

    def test_reject_missing_image_new_image_wrong_quota_or_role(self):
        missing = self.candidate.copy()
        del missing['0']
        for changed in (missing, {**missing, '10': 'branch'}, {**self.candidate, '0': 'fusion'},
                        {**self.candidate, '0': 'both'}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_reassigned_roles(self.historical, changed)
