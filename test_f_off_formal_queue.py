import unittest
import os
import subprocess
import sys

from f_off_formal_queue import validate_rows, terminate_process_group
from f_off_formal_manifest import expected_keys


class FormalQueueTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'process groups are a Linux execution gate')
    def test_failure_terminates_inflight_sibling_group(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                   start_new_session=True)
        try:
            terminate_process_group(process)
            self.assertNotEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

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
