import unittest

from f_off_formal_queue import validate_rows
from f_off_formal_manifest import expected_keys


class FormalQueueTests(unittest.TestCase):
    def test_queue_requires_exact_nine_unique_jobs_and_gpu_binding(self):
        rows = [{'job_key': key, 'physical_gpu': index % 2}
                for index, key in enumerate(sorted(expected_keys()))]
        validate_rows(rows)
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1])
        altered = [dict(row) for row in rows]
        altered[0]['job_key'] = altered[1]['job_key']
        with self.assertRaises(ValueError):
            validate_rows(altered)
        altered = [dict(row) for row in rows]
        altered[0]['physical_gpu'] = 2
        with self.assertRaises(ValueError):
            validate_rows(altered)


if __name__ == '__main__':
    unittest.main()
