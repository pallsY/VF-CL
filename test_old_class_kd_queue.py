import unittest

from run_old_class_kd_queue import expected_cell_keys


class OldClassKDQueueTests(unittest.TestCase):
    def test_pair_plan_requires_eight_unique_cells(self):
        cells = [
            {'dataset': dataset, 'seed': seed, 'variant': variant}
            for dataset in ('isolet', 'upmc_food101')
            for seed in (56, 57)
            for variant in ('oldonly20', 'fullwidth20')
        ]
        self.assertEqual(expected_cell_keys(cells), {
            (cell['dataset'], cell['seed'], cell['variant']) for cell in cells
        })
        with self.assertRaisesRegex(ValueError, 'eight unique'):
            expected_cell_keys(cells[:-1] + [cells[0]])


if __name__ == '__main__':
    unittest.main()
