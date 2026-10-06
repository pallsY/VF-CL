import copy
import unittest

import torch

from factorized_head import factorize_task_probabilities
from models import TopModel


class FactorizedHeadTests(unittest.TestCase):
    def test_composition_preserves_task_mass_and_pre_head_within_task_order(self):
        mixed = torch.tensor([[0.1, 0.2, 0.6, 0.1],
                              [0.3, 0.2, 0.1, 0.4]], dtype=torch.float64).log()
        pre_logits = torch.tensor([[4.0, 1.0, 0.0, 5.0],
                                   [0.0, 3.0, 6.0, 1.0]], dtype=torch.float64)
        task_map = torch.tensor([0, 0, 1, 1])

        result = factorize_task_probabilities(mixed, pre_logits, task_map)

        torch.testing.assert_close(result.exp().sum(1), torch.ones(2, dtype=torch.float64))
        for task in (0, 1):
            columns = (task_map == task).nonzero().flatten()
            torch.testing.assert_close(
                result[:, columns].exp().sum(1), mixed[:, columns].exp().sum(1),
            )
            self.assertTrue(torch.equal(
                result[:, columns].argmax(1), pre_logits[:, columns].argmax(1)
            ))
        self.assertFalse(torch.equal(result, mixed))

    def test_model_readout_round_trips_without_changing_adaptive_forward(self):
        model = TopModel(4, 4)
        with torch.no_grad():
            model.classifier.weight.copy_(8 * torch.eye(4))
            model.classifier.bias.zero_()
        model.set_logit_calibration([0, 1, 2, 3], [1.0] * 4,
                                    [0.0] * 4, [0, 0, 1, 1], 1.0)
        full_weight = model.classifier.weight.detach().clone()
        full_weight[[0, 1]] = full_weight[[1, 0]].clone()
        model.set_adaptive_mixture(full_weight, torch.zeros(4), 0.4,
                                   [0, 1, 2, 3])
        x = torch.eye(4)
        before_state = copy.deepcopy(model.state_dict())
        old_forward = model(x).clone()

        factorized = model.factorized_log_probabilities(x)

        torch.testing.assert_close(model(x), old_forward, rtol=0, atol=0)
        for name, tensor in before_state.items():
            self.assertTrue(torch.equal(model.state_dict()[name], tensor), name)
        restored = TopModel(4, 4)
        restored.load_state_dict(before_state, strict=True)
        torch.testing.assert_close(
            restored.factorized_log_probabilities(x), factorized,
            rtol=0, atol=0,
        )
        task_map = torch.tensor([0, 0, 1, 1])
        for task in (0, 1):
            columns = (task_map == task).nonzero().flatten()
            self.assertTrue(torch.equal(
                factorized[:, columns].argmax(1),
                model._bias_logits(x)[:, columns].argmax(1),
            ))

    def test_rejects_incomplete_task_map(self):
        with self.assertRaises(ValueError):
            factorize_task_probabilities(
                torch.full((1, 4), -1.3862943611198906, dtype=torch.float64),
                torch.zeros((1, 4)), torch.tensor([0, 0, 1, -1]),
            )


if __name__ == '__main__':
    unittest.main()
