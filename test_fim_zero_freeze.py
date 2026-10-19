import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cl_methods.proto_evolve import ProtoEvolveCL


class FimZeroFreezeTests(unittest.TestCase):
    def test_zero_fraction_freezes_no_bottom_parameters(self):
        trainer = SimpleNamespace(
            bottoms=[torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)],
            top_model=torch.nn.Linear(6, 2),
            _aggregate=lambda parts: torch.cat(parts, dim=1),
        )
        method = SimpleNamespace(
            trainer=trainer,
            args=SimpleNamespace(device='cpu', num_parties=2,
                                 fim_freeze_frac=0.0),
            fim_masks=[{}, {}],
        )
        batch = torch.tensor([
            [1., 0., 0., 1.], [0., 1., 1., 0.],
            [1., 1., 0., 0.], [0., 0., 1., 1.],
        ])
        labels = torch.tensor([0, 1, 0, 1])
        with patch('cl_methods.proto_evolve.split_features',
                   side_effect=lambda x, _: [x[:, :2], x[:, 2:]]):
            ProtoEvolveCL._compute_fim(method, [(batch, labels)], 0)
        self.assertFalse(any(value for mask in method.fim_masks
                             for value in mask.values()))
        self.assertTrue(all(parameter.requires_grad
                            for bottom in trainer.bottoms
                            for parameter in bottom.parameters()))


if __name__ == '__main__':
    unittest.main()
