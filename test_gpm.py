import unittest
from unittest.mock import patch

import torch

from cl_methods.gpm import _stable_svd


class StableSVDTests(unittest.TestCase):
    def test_retries_float32_convergence_failure_in_float64(self):
        matrix = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
        original = torch.linalg.svd
        dtypes = []

        def fail_float32(value, **kwargs):
            dtypes.append(value.dtype)
            if value.dtype == torch.float32:
                raise torch._C._LinAlgError("forced float32 failure")
            return original(value, **kwargs)

        with patch("cl_methods.gpm.torch.linalg.svd", side_effect=fail_float32):
            u, singular, vh = _stable_svd(matrix)
        self.assertEqual(dtypes, [torch.float32, torch.float64])
        self.assertEqual(
            (u.dtype, singular.dtype, vh.dtype),
            (torch.float32, torch.float32, torch.float32),
        )
        self.assertTrue(torch.allclose(u @ torch.diag(singular) @ vh, matrix))

    def test_rejects_nonfinite_input_before_backend(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            _stable_svd(torch.tensor([[float("nan")]]))


if __name__ == "__main__":
    unittest.main()
