import unittest
from types import SimpleNamespace

import torch

import cl_methods.ewc as ewc_module
from cl_methods.ewc import EWCCL


class EWCStabilityTests(unittest.TestCase):
    def test_ewc_loss_rejects_nonfinite_total_immediately(self):
        args = SimpleNamespace(
            num_parties=1, device='cpu', ewc_lambda=1000.0,
            ewc_fisher_decay=0.9, ewc_fisher_samples=1,
            lwf_ce_newonly=True, feat_distill_weight=0.0,
        )
        trainer = SimpleNamespace(bottoms=[torch.nn.Linear(1, 1)])
        method = EWCCL(trainer, args)
        with self.assertRaisesRegex(FloatingPointError, 'non-finite EWC loss'):
            method._ewc_loss(
                trainer.bottoms, None, torch.zeros(1, 1), torch.zeros(1),
                loss_ce=torch.tensor(float('inf')),
            )

    def test_normalized_fisher_handles_extreme_float64_dynamic_range(self):
        self.assertTrue(hasattr(ewc_module, '_normalize_fisher'))
        result = ewc_module._normalize_fisher([{
            'large': torch.tensor([1e300], dtype=torch.float64),
            'small': torch.tensor([4.0, 1.0], dtype=torch.float64),
        }], n_samples=1)
        values = torch.cat([value.flatten() for value in result[0].values()])
        self.assertEqual(values.dtype, torch.float32)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(values.ge(0).all())
        self.assertAlmostEqual(float(values.sum()), 1.0, places=6)

    def test_normalized_fisher_rejects_nonfinite_input(self):
        self.assertTrue(hasattr(ewc_module, '_normalize_fisher'))
        with self.assertRaisesRegex(FloatingPointError, 'party 0'):
            ewc_module._normalize_fisher([{
                'weight': torch.tensor([float('inf')], dtype=torch.float64),
            }], n_samples=1)


if __name__ == '__main__':
    unittest.main()
