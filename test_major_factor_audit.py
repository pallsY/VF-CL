import unittest

from major_factor_audit import summarize_core


class MajorFactorAuditTests(unittest.TestCase):
    def test_reports_shared_controls_and_actual_access_difference(self):
        rows = [
            {'dataset': 'isolet', 'method': 'er', 'seed': 42,
             'data_hashes': {'x': 'abc'}, 'num_parties': 4,
             'model_type': 'mlp', 'epochs_per_task': 50,
             'current_train_loader_sample_reads': 100,
             'validation_examples': 0, 'calibration_examples': 0},
            {'dataset': 'isolet', 'method': 'original_adaptive', 'seed': 42,
             'data_hashes': {'x': 'abc'}, 'num_parties': 4,
             'model_type': 'mlp', 'epochs_per_task': 50,
             'current_train_loader_sample_reads': 100,
             'validation_examples': 1040, 'calibration_examples': 0},
        ]
        result = summarize_core(
            rows, ('data_hashes', 'num_parties', 'model_type',
                   'epochs_per_task', 'current_train_loader_sample_reads'))
        self.assertEqual(result['isolet']['mismatches'], {})
        self.assertEqual(result['isolet']['common']['num_parties'], 4)
        self.assertEqual(
            result['isolet']['method_access']['original_adaptive'], [1040])

    def test_marks_a_real_core_difference(self):
        rows = [
            {'dataset': 'upmc_food101', 'method': 'er', 'seed': 42,
             'data_hashes': {'x': 'abc'}, 'validation_examples': 0,
             'calibration_examples': 0},
            {'dataset': 'upmc_food101', 'method': 'original_adaptive',
             'seed': 42, 'data_hashes': {'x': 'changed'},
             'validation_examples': 0, 'calibration_examples': 0},
        ]
        result = summarize_core(rows, ('data_hashes',))
        self.assertIn('data_hashes', result['upmc_food101']['mismatches'])


if __name__ == '__main__':
    unittest.main()
