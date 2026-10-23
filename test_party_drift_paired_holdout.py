import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibration_split import build_manifest, manifest_indices
import run_resource_lite_pilot as pilot


class PairedHoldoutTests(unittest.TestCase):
    def test_explicit_candidate_validation_is_rebuilt_and_excluded(self):
        self.assertTrue(hasattr(pilot, 'validated_paired_manifest'))
        labels = np.repeat(np.arange(3), 100)
        bic = build_manifest(labels, 10, 3)
        validation = build_manifest(labels, 10, 4,
                                    excluded_indices=manifest_indices(bic))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.json'
            path.write_text(json.dumps(validation), encoding='utf-8')
            checked, indices = pilot.validated_paired_manifest(
                labels, path, manifest_indices(bic))
            self.assertEqual(checked, validation)
            self.assertEqual(indices, manifest_indices(validation))
            self.assertFalse(indices & manifest_indices(bic))
            validation['ordered_indices'][0] += 1
            path.write_text(json.dumps(validation), encoding='utf-8')
            with self.assertRaises(ValueError):
                pilot.validated_paired_manifest(labels, path,
                                                manifest_indices(bic))


if __name__ == '__main__':
    unittest.main()
