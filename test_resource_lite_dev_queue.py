import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_resource_lite_dev_queue import (
    ALLOWED_PAIR_DIFFERENCES, atomic_new_json, gpu_is_idle,
    validate_plan,
)


class ResourceLiteDevQueueTests(unittest.TestCase):
    def test_gpu_must_have_no_compute_app_and_low_memory(self):
        rows = '0, GPU-zero, 200, 0\n1, GPU-one, 250, 0\n'
        with mock.patch('subprocess.check_output', side_effect=[rows, '']):
            self.assertTrue(gpu_is_idle(0))
        with mock.patch('subprocess.check_output',
                        side_effect=[rows, 'GPU-zero\n']):
            self.assertFalse(gpu_is_idle(0))
        with mock.patch('subprocess.check_output', side_effect=[
                '0, GPU-zero, 900, 0\n', '']):
            self.assertFalse(gpu_is_idle(0))

    def test_plan_rejects_config_drift_and_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dataset in ('isolet', 'upmc_food101'):
                for seed in (54, 55):
                    common = {
                        'pilot_protocol': 'resource-lite-train-holdout-v1',
                        'pilot_smoke': False,
                        'pilot_dataset': dataset,
                        'seed': seed,
                        'num_parties': 4,
                        'unchanged': 1,
                    }
                    for variant in ('lite18', 'reference20'):
                        config = {
                            **common, 'pilot_variant': variant,
                            'output_dir': str(root / f'{dataset}_seed{seed}_{variant}'),
                        }
                        for field in ALLOWED_PAIR_DIFFERENCES - {
                                'pilot_variant', 'output_dir'}:
                            config[field] = variant
                        (root / f'{dataset}_seed{seed}_{variant}.config.json').write_text(
                            json.dumps(config), encoding='utf-8')
            validate_plan(root)
            altered = root / 'isolet_seed54_lite18.config.json'
            config = json.loads(altered.read_text())
            config['unchanged'] = 2
            altered.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'differences'):
                validate_plan(root)
            config['unchanged'] = 1
            altered.write_text(json.dumps(config))
            (root / 'isolet_seed54_lite18').mkdir()
            with self.assertRaisesRegex(ValueError, 'output'):
                validate_plan(root)

    def test_lock_is_create_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'QUEUE_LOCK.json'
            atomic_new_json(path, {'status': 'started'})
            with self.assertRaises(FileExistsError):
                atomic_new_json(path, {'status': 'duplicate'})


if __name__ == '__main__':
    unittest.main()
