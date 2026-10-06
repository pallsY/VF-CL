"""Generated CIFAR images must retain identity after training augmentation."""
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

import three_dataset_formal_driver as driver


class GeneratedCifarIdentityTests(unittest.TestCase):
    def test_full_fixture_central_crops_are_unique_across_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'fixtures').mkdir()
            with mock.patch.dict(os.environ, {
                    'VFCL_EXPERIMENT_PROFILE': 'full-public-matrix'}):
                driver._generated_image_fixture(root)
            with (root / 'fixtures/image/cifar-100-python/train').open('rb') as stream:
                rows = pickle.load(stream)['data']
            crops = np.ascontiguousarray(
                rows.reshape(10000, 3, 32, 32)[:, :, 4:28, 4:28]
            )
            identities = {
                hashlib.sha256(crop.tobytes()).digest() for crop in crops
            }
            self.assertEqual(10000, len(identities))


if __name__ == '__main__':
    unittest.main()
