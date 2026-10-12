import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from cl_methods.proto_evolve import ProtoEvolveCL
import runner


class ReplayFactorialLossTests(unittest.TestCase):
    def method(self, enabled):
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        method.args = SimpleNamespace(device="cpu")
        method.lambda_A = 0.05
        method.proto_replay_ratio = 1.0
        method.proto_replay_loss_norm = "sample_mean"
        method.party_proto_enabled = False
        method.global_protos = {
            0: {"mean": torch.tensor([1.0, 0.0]),
                "std": torch.tensor([0.1, 0.1])},
            1: {"mean": torch.tensor([0.0, 1.0]),
                "std": torch.tensor([0.1, 0.1])},
        }
        method._method_np_rng = np.random.default_rng(54)
        method._method_torch_rng = torch.Generator().manual_seed(54)
        method._old_top = None
        method.proto_replay_gradient_enabled = enabled
        return method

    def gradients(self, enabled):
        method = self.method(enabled)
        torch.manual_seed(54)
        top = torch.nn.Linear(2, 2)
        ce = torch.tensor(2.0, requires_grad=True)
        loss = method._proto_loss([], top, torch.zeros(2, 2),
                                  torch.tensor([0, 1]), loss_ce=ce)
        loss.backward()
        grad = (None if top.weight.grad is None
                else top.weight.grad.detach().clone())
        return ce.grad.item(), grad, (
            copy.deepcopy(method._method_np_rng.bit_generator.state),
            method._method_torch_rng.get_state().clone(),
        )

    def test_switch_removes_only_prototype_gradient_and_preserves_rng(self):
        on_ce, on_top, on_rng = self.gradients(True)
        off_ce, off_top, off_rng = self.gradients(False)
        self.assertAlmostEqual(on_ce, 0.95, places=6)
        self.assertAlmostEqual(off_ce, 0.95, places=6)
        self.assertGreater(on_top.norm().item(), 0.0)
        self.assertIsNone(off_top)
        self.assertEqual(on_rng[0], off_rng[0])
        self.assertTrue(torch.equal(on_rng[1], off_rng[1]))

    def test_disabled_switch_is_bound_to_checkpoint_protocol(self):
        args = SimpleNamespace(seed=54, data="tabvfl", cl_method="proto_evolve",
                               num_tasks=13, num_parties=4,
                               head_consolidation_enabled=1,
                               head_consolidation_mode="adaptive_dual_branch",
                               head_consolidation_samples_per_class=20,
                               proto_replay_gradient_enabled=0,
                               formal_deferred_evaluation=False)
        disabled = runner._checkpoint_protocol(args)
        self.assertIs(disabled["proto_replay_gradient_enabled"], False)
        args.proto_replay_gradient_enabled = 1
        self.assertNotIn("proto_replay_gradient_enabled",
                         runner._checkpoint_protocol(args))
        args.head_consolidation_mode = "full_classifier"
        args.head_consolidation_samples_per_class = 40
        args.proto_replay_gradient_enabled = 0
        fixed = runner._checkpoint_protocol(args)
        self.assertEqual(fixed["head_consolidation_samples_per_class"], 40)
        self.assertIs(fixed["proto_replay_gradient_enabled"], False)


if __name__ == "__main__":
    unittest.main()
