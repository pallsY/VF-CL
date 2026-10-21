import unittest

from verify_sampleonly_replay_identity import first_two_tasks


class IdentityTaskTests(unittest.TestCase):
    def test_isolet_and_upmc_task_boundaries(self):
        self.assertEqual(first_two_tasks('0,1|2,3|4,5'), ([0, 1], [2, 3]))
        first, second = first_two_tasks(
            '0,1,2,3,4,5,6,7,8,9,10|11,12,13,14,15,16,17,18,19,20')
        self.assertEqual(len(first), 11)
        self.assertEqual(len(second), 10)
        self.assertEqual(second[0], 11)


if __name__ == '__main__':
    unittest.main()
