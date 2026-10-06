import os
import random
import unittest

import numpy as np
import torch

from determinism import configure_determinism, derive_seed, seed_worker, tensor_sha256


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.old_env = {
            key: os.environ.get(key)
            for key in (
                'PYTHONHASHSEED',
                'CUBLAS_WORKSPACE_CONFIG',
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
            )
        }
        os.environ.update({
            'PYTHONHASHSEED': '42',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        })

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_derive_seed_is_stable_and_namespaced(self):
        self.assertEqual(derive_seed(42, 'data', 1), derive_seed(42, 'data', 1))
        self.assertNotEqual(derive_seed(42, 'data', 1), derive_seed(42, 'method', 1))
        self.assertGreaterEqual(derive_seed(42, 'data', 1), 0)
        self.assertLess(derive_seed(42, 'data', 1), 2**32)

    def test_configure_determinism_replays_global_rngs_and_sets_backends(self):
        configure_determinism(42)
        first = (random.random(), np.random.rand(), torch.rand(3))
        configure_determinism(42)
        second = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_configure_determinism_rejects_missing_prestart_environment(self):
        os.environ.pop('OMP_NUM_THREADS')
        with self.assertRaisesRegex(RuntimeError, 'OMP_NUM_THREADS'):
            configure_determinism(42)

    def test_seed_worker_seeds_python_numpy_and_torch_from_torch_initial_seed(self):
        torch.manual_seed(1234)
        seed_worker(0)
        first = (random.random(), np.random.rand(), torch.rand(3))
        torch.manual_seed(1234)
        seed_worker(0)
        second = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_tensor_sha256_is_exact_and_cpu_only(self):
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertEqual(tensor_sha256(x), tensor_sha256(x.clone()))
        y = x.clone()
        y[0, 0] = 1
        self.assertNotEqual(tensor_sha256(x), tensor_sha256(y))
        if torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, 'CPU'):
                tensor_sha256(x.cuda())


if __name__ == '__main__':
    unittest.main()
