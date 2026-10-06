import unittest

import torch
import torch.nn.functional as F

from adaptive_head_consolidation import mix_log_probabilities
from models import TopModel


class AdaptiveTopModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.x = torch.tensor([
            [1.0, -2.0, 0.5],
            [-0.25, 0.75, 2.0],
        ])
        self.full_weight = torch.tensor([
            [0.5, -0.25, 1.0],
            [-1.0, 0.75, 0.25],
            [0.2, 0.4, -0.8],
        ])
        self.full_bias = torch.tensor([0.3, -0.4, 0.2])
        self.classes = [0, 2, 3]

    def _linear_model(self):
        model = TopModel(3, 4, cosine=False)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([
                [0.1, 0.2, -0.3],
                [-0.4, 0.5, 0.6],
                [0.7, -0.8, 0.9],
                [-1.0, 1.1, -1.2],
            ]))
            model.classifier.bias.copy_(torch.tensor([0.2, -0.1, 0.4, -0.3]))
        return model

    def _install(self, model, gate=0.35):
        model.set_adaptive_mixture(
            self.full_weight, self.full_bias, gate, self.classes
        )
        return model

    def _calibrated_install(self):
        model = self._linear_model()
        model.set_logit_calibration(
            self.classes, [1.1, 0.9, 1.2], [0.1, -0.2, 0.3],
            [0, 1, 1], 1.3,
        )
        return self._install(model)

    def test_disabled_forward_preserves_linear_and_cosine_logits_exactly(self):
        linear = self._linear_model()
        expected_linear = linear.classifier(self.x)
        torch.testing.assert_close(linear(self.x), expected_linear, rtol=0, atol=0)

        cosine = TopModel(3, 4, cosine=True, scale=7.0)
        expected_cosine = cosine.scale * F.linear(
            F.normalize(self.x, dim=1),
            F.normalize(cosine.classifier.weight, dim=1),
        )
        torch.testing.assert_close(cosine(self.x), expected_cosine, rtol=0, atol=0)

        self._install(linear)
        linear.clear_adaptive_mixture()
        torch.testing.assert_close(linear(self.x), expected_linear, rtol=0, atol=0)

    def test_linear_branches_align_over_ordered_classes_and_mix_exactly(self):
        model = self._linear_model()
        model.set_logit_calibration(
            self.classes,
            [1.1, 1.2, 0.8],
            [0.1, 0.3, -0.4],
            [0, 1, 1],
            1.3,
        )
        self._install(model)

        log_p_full, log_p_bias = model.branch_log_probabilities(self.x)
        expected_full = F.log_softmax(
            F.linear(self.x, self.full_weight, self.full_bias).double(), dim=1
        )
        bias_logits = model._apply_logit_calibration(model.classifier(self.x))
        expected_bias = F.log_softmax(
            bias_logits[:, self.classes].double(), dim=1
        )

        self.assertEqual(log_p_full.dtype, torch.float64)
        self.assertEqual(log_p_bias.dtype, torch.float64)
        torch.testing.assert_close(log_p_full, expected_full, rtol=0, atol=0)
        torch.testing.assert_close(log_p_bias, expected_bias, rtol=0, atol=0)
        torch.testing.assert_close(
            model(self.x), mix_log_probabilities(expected_full, expected_bias, 0.35),
            rtol=0, atol=0,
        )

    def test_cosine_branches_share_normalization_scale_and_ignore_biases(self):
        model = TopModel(3, 4, cosine=True, scale=5.5)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([
                [0.1, 0.2, -0.3],
                [-0.4, 0.5, 0.6],
                [0.7, -0.8, 0.9],
                [-1.0, 1.1, -1.2],
            ]))
            model.classifier.bias.fill_(1000.0)
        huge_full_bias = torch.full_like(self.full_bias, -1000.0)
        model.set_adaptive_mixture(
            self.full_weight, huge_full_bias, 0.4, self.classes
        )

        log_p_full, log_p_bias = model.branch_log_probabilities(self.x)
        expected_full = F.log_softmax((
            model.scale * F.linear(
                F.normalize(self.x, dim=1),
                F.normalize(self.full_weight, dim=1),
            )
        ).double(), dim=1)
        expected_bias = F.log_softmax((
            model.scale * F.linear(
                F.normalize(self.x, dim=1),
                F.normalize(model.classifier.weight, dim=1),
            )[:, self.classes]
        ).double(), dim=1)

        torch.testing.assert_close(log_p_full, expected_full, rtol=0, atol=0)
        torch.testing.assert_close(log_p_bias, expected_bias, rtol=0, atol=0)

    def test_exact_gate_endpoints_return_the_selected_branch(self):
        model = self._linear_model()
        for gate, selected in ((0.0, 1), (1.0, 0)):
            with self.subTest(gate=gate):
                model.set_adaptive_mixture(
                    self.full_weight, self.full_bias, gate, self.classes
                )
                branches = model.branch_log_probabilities(self.x)
                actual = model(self.x)
                self.assertTrue(torch.equal(actual, branches[selected]))
                self.assertNotEqual(actual.data_ptr(), branches[selected].data_ptr())

    def test_sparse_fixed_endpoints_round_trip_without_duplicate_classifier(self):
        for gate in (0.0, 1.0):
            with self.subTest(gate=gate):
                model = self._linear_model()
                if gate == 0.0:
                    model.set_logit_calibration(
                        self.classes, [1.1, 0.9, 1.2],
                        [0.1, -0.2, 0.3], [5, 5, 7], 1.3,
                    )
                    logits = model._apply_logit_calibration(
                        model.classifier(self.x)
                    )[:, self.classes]
                else:
                    logits = model.classifier(self.x)[:, self.classes]
                expected = F.log_softmax(logits.double(), dim=1)
                setter = getattr(model, 'set_adaptive_endpoint', None)
                self.assertIsNotNone(setter, 'sparse endpoint API is missing')
                setter(gate, self.classes)
                self.assertTrue(bool(model._adaptive_enabled))
                self.assertEqual(float(model._adaptive_gate), gate)
                self.assertEqual(model._adaptive_full_weight.numel(), 0)
                self.assertEqual(model._adaptive_full_bias.numel(), 0)
                torch.testing.assert_close(model(self.x), expected, rtol=0, atol=0)
                state = model.state_dict()
                restored = self._linear_model()
                restored.load_state_dict(state, strict=True)
                self.assertEqual(restored._adaptive_full_weight.numel(), 0)
                torch.testing.assert_close(
                    restored(self.x), expected, rtol=0, atol=0,
                )

    def test_set_copies_inputs_and_invalid_update_is_atomic(self):
        model = self._install(self._linear_model())
        expected = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if '_adaptive_' in key
        }
        self.full_weight.zero_()
        self.full_bias.zero_()
        torch.testing.assert_close(model._adaptive_full_weight, expected['_adaptive_full_weight'])
        torch.testing.assert_close(model._adaptive_full_bias, expected['_adaptive_full_bias'])

        with self.assertRaises(ValueError):
            model.set_adaptive_mixture(
                torch.full((3, 3), float('nan')), self.full_bias,
                0.5, self.classes,
            )
        for key, value in expected.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_set_rejects_values_that_overflow_classifier_dtype_atomically(self):
        limit = float(torch.finfo(torch.float32).max) * 2.0
        for field in ('weight', 'bias'):
            model = self._install(self._linear_model())
            expected = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
                if '_adaptive_' in key
            }
            weight = self.full_weight.double()
            bias = self.full_bias.double()
            if field == 'weight':
                weight = torch.full((3, 3), limit, dtype=torch.float64)
            else:
                bias = torch.full((3,), limit, dtype=torch.float64)

            with self.subTest(field=field), self.assertRaises(ValueError):
                model.set_adaptive_mixture(
                    weight, bias, 0.5, self.classes
                )
            for key, value in expected.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_module_dtype_casts_preserve_float64_gate_and_reload(self):
        cases = (
            ('float', lambda model: model.float(), lambda: TopModel(3, 4).float()),
            ('half', lambda model: model.half(), lambda: TopModel(3, 4).half()),
            ('to_float32', lambda model: model.to(dtype=torch.float32),
             lambda: TopModel(3, 4).double().to(dtype=torch.float32)),
        )
        expected_gate = torch.tensor(0.35, dtype=torch.float64)
        for name, cast, restored_factory in cases:
            with self.subTest(cast=name):
                source = self._linear_model()
                if name == 'to_float32':
                    source = source.double()
                self._install(source)
                cast(source)

                self.assertEqual(source._adaptive_gate.dtype, torch.float64)
                self.assertTrue(torch.equal(source._adaptive_gate.cpu(), expected_gate))
                restored = restored_factory()
                restored.load_state_dict(source.state_dict())
                self.assertEqual(restored._adaptive_gate.dtype, torch.float64)
                self.assertTrue(torch.equal(
                    restored._adaptive_gate.cpu(), expected_gate
                ))

    def test_to_empty_accepts_recurse_and_preserves_available_gate(self):
        model = self._install(self._linear_model())
        expected_gate = model._adaptive_gate.detach().clone()

        returned = model.to_empty(device='cpu')

        self.assertIs(returned, model)
        self.assertEqual(model._adaptive_gate.dtype, torch.float64)
        self.assertTrue(torch.equal(model._adaptive_gate, expected_gate))

    def test_nested_meta_to_empty_materializes_then_reloads_exact_state(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, top):
                super().__init__()
                self.top = top

        source = self._install(self._linear_model(), gate=0.625)
        state = source.state_dict()
        expected_output = source(self.x)
        wrapper = Wrapper(TopModel(3, 4, cosine=False)).to(device='meta')

        wrapper.to_empty(device='cpu')
        wrapper.top.load_state_dict(state)

        self.assertEqual(wrapper.top._adaptive_gate.dtype, torch.float64)
        self.assertTrue(torch.equal(
            wrapper.top._adaptive_gate, state['_adaptive_gate']
        ))
        self.assertTrue(torch.equal(wrapper.top(self.x), expected_output))

    def test_adaptive_state_dict_round_trip_resizes_fresh_empty_buffers(self):
        source = self._install(self._linear_model(), gate=0.625)
        before_branches = source.branch_log_probabilities(self.x)
        before_output = source(self.x)
        state = source.state_dict()

        restored = TopModel(3, 4, cosine=False)
        self.assertEqual(restored._adaptive_full_weight.numel(), 0)
        restored.load_state_dict(state)

        for key, value in state.items():
            self.assertTrue(torch.equal(restored.state_dict()[key], value), key)
        after_branches = restored.branch_log_probabilities(self.x)
        for before, after in zip(before_branches, after_branches):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(before_output, restored(self.x)))

    def test_calibrated_adaptive_state_round_trips_exactly(self):
        source = self._calibrated_install()
        state = source.state_dict()
        expected_branches = source.branch_log_probabilities(self.x)
        expected_output = source(self.x)

        restored = TopModel(3, 4, cosine=False)
        restored.load_state_dict(state)

        for before, after in zip(
                expected_branches, restored.branch_log_probabilities(self.x)):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(expected_output, restored(self.x)))

    def test_set_rejects_live_bias_class_order_mismatch_before_install(self):
        model = self._linear_model()
        model.set_logit_calibration(
            [0, 1, 2, 3], [1.0] * 4, [0.0] * 4, [0, 0, 1, 1], 1.3
        )

        with self.assertRaises(ValueError):
            model.set_adaptive_mixture(
                self.full_weight, self.full_bias, 0.5, self.classes
            )

        self.assertFalse(bool(model._adaptive_enabled))
        self.assertEqual(model._adaptive_full_weight.numel(), 0)

    def test_enabled_state_load_resizes_smaller_and_larger_live_buffers(self):
        large = self._install(self._linear_model(), gate=0.625)
        small = self._linear_model()
        small.set_adaptive_mixture(
            self.full_weight[[0, 2]], self.full_bias[[0, 2]],
            0.25, [0, 3],
        )

        for source, restored in (
                (small, self._install(self._linear_model())),
                (large, small)):
            with self.subTest(classes=source._adaptive_class_order.tolist()):
                expected = source(self.x)
                state = source.state_dict()
                restored.load_state_dict(state)
                for key, value in state.items():
                    self.assertTrue(torch.equal(restored.state_dict()[key], value), key)
                self.assertTrue(torch.equal(restored(self.x), expected))

    def test_legacy_checkpoint_without_any_adaptive_fields_loads_disabled(self):
        source = self._linear_model()
        legacy = {
            key: value for key, value in source.state_dict().items()
            if '_adaptive_' not in key
        }
        restored = TopModel(3, 4, cosine=False)

        restored.load_state_dict(legacy)

        self.assertFalse(bool(restored._adaptive_enabled))
        torch.testing.assert_close(
            restored(self.x), restored.classifier(self.x), rtol=0, atol=0
        )

    def test_meta_to_empty_legacy_load_uses_canonical_calibration_defaults(self):
        source = self._linear_model()
        legacy = {
            key: value for key, value in source.state_dict().items()
            if '_adaptive_' not in key and '_logit_calibration_' not in key
        }
        expected = source.classifier(self.x)
        restored = TopModel(3, 4, cosine=False).to(device='meta')
        restored.to_empty(device='cpu')
        with torch.no_grad():
            restored._logit_calibration_alpha.fill_(17.0)
            restored._logit_calibration_bias.fill_(-4.0)
            restored._logit_calibration_task.fill_(3)
            restored._logit_calibration_task_weight.zero_()
            restored._logit_calibration_enabled.fill_(True)

        restored.load_state_dict(legacy)

        self.assertTrue(torch.equal(
            restored._logit_calibration_alpha, torch.ones(4)
        ))
        self.assertTrue(torch.equal(
            restored._logit_calibration_bias, torch.zeros(4)
        ))
        self.assertTrue(torch.equal(
            restored._logit_calibration_task,
            torch.full((4,), -1, dtype=torch.long),
        ))
        self.assertTrue(torch.equal(
            restored._logit_calibration_task_weight, torch.tensor(1.0)
        ))
        self.assertTrue(torch.equal(
            restored._logit_calibration_enabled, torch.tensor(False)
        ))
        torch.testing.assert_close(restored(self.x), expected, rtol=0, atol=0)

    def test_legacy_checkpoint_clears_previously_enabled_adaptive_state(self):
        source = self._linear_model()
        legacy = {
            key: value for key, value in source.state_dict().items()
            if '_adaptive_' not in key
        }
        restored = self._install(self._linear_model())

        restored.load_state_dict(legacy)

        self.assertFalse(bool(restored._adaptive_enabled))
        self.assertEqual(restored._adaptive_full_weight.numel(), 0)
        torch.testing.assert_close(
            restored(self.x), restored.classifier(self.x), rtol=0, atol=0
        )

    def test_disabled_versioned_checkpoint_clears_enabled_dynamic_buffers(self):
        disabled_state = self._linear_model().state_dict()
        restored = self._install(self._linear_model())

        restored.load_state_dict(disabled_state)

        self.assertFalse(bool(restored._adaptive_enabled))
        self.assertEqual(restored._adaptive_full_weight.shape, torch.Size([0, 3]))
        self.assertEqual(restored._adaptive_full_bias.numel(), 0)

    def test_partial_or_malformed_enabled_checkpoint_hard_fails(self):
        valid = self._install(self._linear_model()).state_dict()
        partial = dict(valid)
        partial.pop('_adaptive_gate')
        malformed = dict(valid)
        malformed['_adaptive_full_bias'] = torch.zeros(2)
        non_tensor = dict(valid)
        non_tensor['_adaptive_class_order'] = [0, 2, 3]
        missing_bias_branch = dict(valid)
        missing_bias_branch.pop('_logit_calibration_alpha')

        for state in (partial, malformed, non_tensor, missing_bias_branch):
            with self.subTest(keys=tuple(sorted(state))):
                with self.assertRaises(RuntimeError):
                    TopModel(3, 4, cosine=False).load_state_dict(state)

    def test_enabled_checkpoint_validates_complete_bias_branch_state(self):
        valid = self._calibrated_install().state_dict()
        invalid_states = []
        for key, value in (
            ('classifier.weight', torch.full((4, 3), float('nan'))),
            ('classifier.bias', torch.full((4,), float('inf'))),
            ('_logit_calibration_alpha', torch.full((4,), float('nan'))),
            ('_logit_calibration_bias', torch.full((4,), float('inf'))),
            ('_logit_calibration_task_weight', torch.tensor(0.0)),
            ('_logit_calibration_task', torch.full((4,), -1, dtype=torch.long)),
            ('_logit_calibration_enabled', torch.tensor(1, dtype=torch.long)),
            ('_logit_calibration_alpha', valid['_logit_calibration_alpha'].double()),
            ('_logit_calibration_task', valid['_logit_calibration_task'].int()),
        ):
            state = dict(valid)
            state[key] = value
            invalid_states.append((key, state))
        for missing in ('classifier.weight', 'classifier.bias'):
            state = dict(valid)
            state.pop(missing)
            invalid_states.append(('missing ' + missing, state))

        for label, state in invalid_states:
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                TopModel(3, 4, cosine=False).load_state_dict(
                    state, strict=False
                )

    def test_enabled_cosine_checkpoint_requires_finite_shared_scale(self):
        source = TopModel(3, 4, cosine=True)
        source.set_adaptive_mixture(
            self.full_weight, self.full_bias, 0.5, self.classes
        )
        valid = source.state_dict()
        missing = dict(valid)
        missing.pop('scale')
        nonfinite = dict(valid)
        nonfinite['scale'] = torch.tensor(float('nan'))

        for state in (missing, nonfinite):
            with self.assertRaises(RuntimeError):
                TopModel(3, 4, cosine=True).load_state_dict(
                    state, strict=False
                )

    def test_enabled_checkpoint_rejects_head_mode_mismatch_non_strict(self):
        linear = self._install(self._linear_model())
        cosine = TopModel(3, 4, cosine=True)
        cosine.set_adaptive_mixture(
            self.full_weight, self.full_bias, 0.5, self.classes
        )

        mismatches = (
            (cosine.state_dict(), TopModel(3, 4, cosine=False)),
            (linear.state_dict(), TopModel(3, 4, cosine=True)),
        )
        for state, restored in mismatches:
            with self.subTest(target_cosine=restored.cosine):
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(state, strict=False)
                self.assertFalse(bool(restored._adaptive_enabled))

    def test_present_disabled_adaptive_payload_requires_canonical_dtypes(self):
        valid = self._linear_model().state_dict()
        replacements = (
            ('_adaptive_version', torch.tensor(0.0)),
            ('_adaptive_gate', torch.tensor(0, dtype=torch.long)),
            ('_adaptive_full_weight', torch.empty(0, 3, dtype=torch.long)),
            ('_adaptive_full_bias', torch.empty(0, dtype=torch.long)),
            ('_adaptive_class_order', torch.empty(0, dtype=torch.float32)),
        )
        for key, value in replacements:
            state = dict(valid)
            state[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                TopModel(3, 4, cosine=False).load_state_dict(state)

    def test_unsupported_version_nonfinite_gate_and_bad_gate_are_rejected(self):
        model = self._linear_model()
        invalid_calls = (
            {'gate': float('nan')},
            {'gate': -0.01},
            {'gate': 1.01},
            {'gate': True},
            {'gate': 0.5, 'version': 2},
        )
        for values in invalid_calls:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                model.set_adaptive_mixture(
                    self.full_weight, self.full_bias,
                    values['gate'], self.classes, values.get('version', 1),
                )

        state = self._install(model).state_dict()
        state['_adaptive_version'] = torch.tensor(2, dtype=torch.long)
        with self.assertRaises(RuntimeError):
            TopModel(3, 4, cosine=False).load_state_dict(state)

    def test_shape_class_order_and_nonfinite_state_are_rejected(self):
        model = self._linear_model()
        invalid = (
            (torch.zeros(3, 2), self.full_bias, self.classes),
            (self.full_weight, torch.zeros(2), self.classes),
            (self.full_weight, self.full_bias, [0, 2]),
            (self.full_weight, self.full_bias, [0, 2, 2]),
            (self.full_weight, self.full_bias, [0, 3, 2]),
            (self.full_weight, self.full_bias, [0, 2, 4]),
            (torch.full((3, 3), float('inf')), self.full_bias, self.classes),
        )
        for weight, bias, classes in invalid:
            with self.subTest(shape=tuple(weight.shape), classes=classes):
                with self.assertRaises((TypeError, ValueError)):
                    model.set_adaptive_mixture(weight, bias, 0.5, classes)

        state = self._install(model).state_dict()
        state['_adaptive_class_order'] = torch.tensor([0, 3, 2])
        with self.assertRaises(RuntimeError):
            TopModel(3, 4, cosine=False).load_state_dict(state)

    def test_branch_probabilities_reject_nonfinite_live_bias_branch(self):
        model = self._install(self._linear_model())
        with torch.no_grad():
            model.classifier.weight[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            model.branch_log_probabilities(self.x)

    def test_linear_bias_branch_preserves_classifier_pre_and_forward_hooks(self):
        model = self._linear_model()
        calls = []

        def pre_hook(module, inputs):
            calls.append(('pre', module))
            return (inputs[0] + 1.0,)

        def forward_hook(module, inputs, output):
            calls.append(('forward', module))
            return output * 2.0

        pre_handle = model.classifier.register_forward_pre_hook(pre_hook)
        forward_handle = model.classifier.register_forward_hook(forward_hook)
        expected_logits = F.linear(
            self.x + 1.0, model.classifier.weight, model.classifier.bias
        ) * 2.0
        try:
            torch.testing.assert_close(
                model(self.x), expected_logits, rtol=0, atol=0
            )
            self.assertEqual([name for name, _ in calls], ['pre', 'forward'])
            self.assertTrue(all(module is model.classifier for _, module in calls))

            calls.clear()
            self._install(model)
            _, log_p_bias = model.branch_log_probabilities(self.x)
            expected_bias = F.log_softmax(
                expected_logits[:, self.classes].double(), dim=1
            )
            torch.testing.assert_close(log_p_bias, expected_bias, rtol=0, atol=0)
            self.assertEqual([name for name, _ in calls], ['pre', 'forward'])
        finally:
            pre_handle.remove()
            forward_handle.remove()

    def test_class_expansion_is_rejected_only_while_adaptive_state_is_enabled(self):
        model = self._install(self._linear_model())
        with self.assertRaises(RuntimeError):
            model.expand_classes(5, 'cpu')

        model.clear_adaptive_mixture()
        model.expand_classes(5, 'cpu')
        self.assertEqual(model.classifier.out_features, 5)


if __name__ == '__main__':
    unittest.main()
