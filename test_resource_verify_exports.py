"""Independent exported-ledger checks reject wrong rows and category sums."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from resource_verify_exports import verify_exports


class VerifyResourceExportsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.memory_json = root / 'memory.json'
        self.memory_csv = root / 'memory.csv'
        self.access_json = root / 'access.json'
        self.access_csv = root / 'access.csv'
        self.memory = [{
            'job': 'isolet/er/42', 'final_model_bytes': 40,
            'final_raw_bytes': 24, 'final_logits_bytes': 0,
            'final_feature_bytes': 0, 'final_teacher_bytes': 0,
            'final_other_bytes': 0, 'final_audit_only_bytes': 99,
            'final_learning_bytes': 64, 'max_learning_bytes': 64,
            'final_raw_examples': 2, 'max_raw_examples': 2,
            'max_raw_bytes': 24,
        }]
        self.access = [{
            'job': 'isolet/er/42', 'nominal_optimizer_steps': 5,
            'head_candidate_steps': 0, 'validation_examples': 0,
            'calibration_examples': 0, 'test_accesses': 1,
            'flops_measured': False,
        }]
        self.write()

    def write(self):
        for data, json_path, csv_path in (
                (self.memory, self.memory_json, self.memory_csv),
                (self.access, self.access_json, self.access_csv)):
            json_path.write_text(json.dumps(data))
            with csv_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    def verify(self):
        return verify_exports(
            self.memory_json, self.memory_csv,
            self.access_json, self.access_csv,
            expected_jobs={'isolet/er/42'})

    def test_exact_rows_and_audit_bytes_excluded(self):
        result = self.verify()
        self.assertEqual(result['verified_cells'], 1)

    def test_bad_category_sum_rejected(self):
        self.memory[0]['final_learning_bytes'] = 163
        self.write()
        with self.assertRaisesRegex(ValueError, 'category'):
            self.verify()

    def test_csv_mutation_and_missing_access_rejected(self):
        with self.memory_csv.open('a', encoding='utf-8') as handle:
            handle.write('junk,row\n')
        with self.assertRaisesRegex(ValueError, 'CSV'):
            self.verify()
        self.write()
        self.access[0]['job'] = 'isolet/er/43'
        self.write()
        with self.assertRaisesRegex(ValueError, 'job'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
