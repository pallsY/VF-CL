import tempfile
import unittest
from pathlib import Path

from run_sampleonly_fim_queue import gpu_idle_from_rows, prepare_root


class FimQueueGuardsTests(unittest.TestCase):
    def test_refuses_second_queue_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'ablation'
            prepare_root(root, 'frozen-commit')
            self.assertTrue((root / 'QUEUE_LOCK.json').is_file())
            with self.assertRaises(FileExistsError):
                prepare_root(root, 'frozen-commit')

    def test_gpu_idle_checks_uuid_and_memory(self):
        devices = ['0, GPU-zero, 35, 0', '1, GPU-one, 9500, 80']
        apps = ['GPU-one, 1234']
        self.assertTrue(gpu_idle_from_rows(devices, apps, 0))
        self.assertFalse(gpu_idle_from_rows(devices, apps, 1))


if __name__ == '__main__':
    unittest.main()
