import contextlib
import io
import unittest
from types import SimpleNamespace

import torch

from cl_methods.proto_evolve import ProtoEvolveCL


class FIMApplicationTests(unittest.TestCase):
    def _method(self, enabled):
        method = ProtoEvolveCL.__new__(ProtoEvolveCL)
        bottom = torch.nn.Linear(2, 2)
        method.args = SimpleNamespace(num_parties=1, fim_apply_mask=enabled,
                                      device='cpu')
        method.trainer = SimpleNamespace(
            bottoms=[bottom],
            top_model=SimpleNamespace(expand_classes=lambda *_: None),
        )
        method.dep_tracking_enabled = False
        method.current_task_classes = []
        method.fim_masks = [{'weight': True, 'bias': False}]
        method._reset_method_rng = lambda _: None
        return method, bottom

    def test_off_preserves_computed_mask_but_applies_zero_freeze(self):
        method, bottom = self._method(0)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            method.before_task(1, [2], [0, 1])
        self.assertEqual(method.fim_masks, [{'weight': True, 'bias': False}])
        self.assertTrue(all(p.requires_grad for p in bottom.parameters()))
        self.assertIn('fraction=0.000000', output.getvalue())

    def test_default_on_keeps_original_freeze_behavior(self):
        method, bottom = self._method(1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            method.before_task(1, [2], [0, 1])
        self.assertFalse(bottom.weight.requires_grad)
        self.assertTrue(bottom.bias.requires_grad)
        self.assertIn('FIM freeze: 1/2 bottom params frozen', output.getvalue())

    def test_off_logs_zero_applied_freeze_on_first_task(self):
        method, bottom = self._method(0)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            method.before_task(0, [0, 1], [])
        self.assertTrue(all(p.requires_grad for p in bottom.parameters()))
        self.assertIn('fraction=0.000000', output.getvalue())


if __name__ == '__main__':
    unittest.main()
