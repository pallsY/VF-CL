import unittest

import torch

from analyze_isolet_prototype_feature_head import (
    merge_raw_synthetic, synthetic_by_class,
)


class PrototypeFeatureHeadTests(unittest.TestCase):
    def test_deterministic_transient_features_and_exact_count(self):
        prototypes = {
            0: {'mean': torch.tensor([1.0, 0.0]), 'std': torch.tensor([0.1, 0.2])},
            1: {'mean': torch.tensor([0.0, 1.0]), 'std': torch.tensor([0.2, 0.1])},
        }
        first = synthetic_by_class(prototypes, 123, 3)
        again = synthetic_by_class(prototypes, 123, 3)
        different = synthetic_by_class(prototypes, 124, 3)
        self.assertEqual(set(first), {0, 1})
        for class_id in (0, 1):
            self.assertEqual(tuple(first[class_id].shape), (3, 2))
            self.assertTrue(torch.equal(first[class_id][0], prototypes[class_id]['mean']))
            self.assertTrue(torch.equal(first[class_id], again[class_id]))
            self.assertFalse(torch.equal(first[class_id][1:], different[class_id][1:]))
        raw = {class_id: torch.ones(2, 2) * class_id for class_id in (0, 1)}
        combined = merge_raw_synthetic(raw, first, 2, 3)
        self.assertEqual({key: value.shape[0] for key, value in combined.items()},
                         {0: 5, 1: 5})
        self.assertTrue(torch.equal(combined[0][:2], raw[0]))
        self.assertEqual(raw[0].shape[0], 2)


if __name__ == '__main__':
    unittest.main()
