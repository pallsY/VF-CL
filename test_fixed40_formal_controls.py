import unittest
from types import SimpleNamespace

import runner


class Fixed40FormalControlTests(unittest.TestCase):
    def test_fixed_heads_bind_40_to_protocol_without_changing_20(self):
        for mode in ('full_classifier', 'task_class_bias'):
            with self.subTest(mode=mode):
                args = SimpleNamespace(
                    seed=42, data='tabvfl', cl_method='proto_evolve',
                    num_tasks=2, num_parties=2,
                    head_consolidation_enabled=1,
                    head_consolidation_mode=mode,
                    head_consolidation_samples_per_class=40,
                    formal_deferred_evaluation=False,
                )
                protocol = runner._checkpoint_protocol(args)
                self.assertEqual(protocol['head_consolidation_samples_per_class'], 40)
                args.head_consolidation_samples_per_class = 20
                old = runner._checkpoint_protocol(args)
                self.assertNotIn('head_consolidation_samples_per_class', old)
                self.assertEqual({key: value for key, value in protocol.items()
                                  if key != 'head_consolidation_samples_per_class'}, old)

    def test_fixed_bias_formal_head_is_installed_only_once(self):
        class Method:
            head_consolidation_mode = 'task_class_bias'

            def __init__(self):
                self.fit_count = 0

            def _consolidate_head(self, task_id):
                self.fit_count += 1
                return {'task_id': task_id}

            def after_task(self, _loader, task_id):
                return self._consolidate_head(task_id)

        class ForbiddenValidation:
            def authorize_formal_access(self, **_kwargs):
                raise AssertionError('fixed Bias accessed late validation')

        method = Method()
        trainer = SimpleNamespace(dataset_ref=None)
        args = SimpleNamespace(
            formal_deferred_evaluation=True,
            cl_method='proto_evolve',
            head_consolidation_enabled=1,
            head_consolidation_mode='task_class_bias',
        )
        runner._call_after_task(method, args, trainer, None, object(), 1)
        self.assertEqual(method.fit_count, 1)
        self.assertIsNone(runner._finalize_formal_internal_state(
            args=args, dataset=ForbiddenValidation(), cl_method=method,
            task_classes={0: [0], 1: [1]}, final_event=1, final_task=1,
        ))
        self.assertEqual(method.fit_count, 1)

    def test_adaptive_remains_the_late_internal_head(self):
        args = SimpleNamespace(
            cl_method='proto_evolve', head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )
        self.assertTrue(runner._is_formal_internal_method(args))

    def test_fixed_heads_need_no_validation_phase_before_formal_test(self):
        dataset = runner.VFLDataset.__new__(runner.VFLDataset)
        for mode in ('full_classifier', 'task_class_bias'):
            with self.subTest(mode=mode):
                dataset.args = SimpleNamespace(
                    head_consolidation_enabled=1,
                    head_consolidation_mode=mode,
                    bic_enabled=0,
                )
                dataset._validate_formal_phase_transition(
                    {'split': 'test'}, [],
                )
        dataset.args.head_consolidation_mode = 'adaptive_dual_branch'
        with self.assertRaisesRegex(ValueError, 'ordered validation'):
            dataset._validate_formal_phase_transition({'split': 'test'}, [])


if __name__ == '__main__':
    unittest.main()
