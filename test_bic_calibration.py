import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from bic_calibration import (
    TaskAffineCalibrator,
    fit_final_calibrator,
    summarize_paired_logits,
)
from metrics import cache_formal_batches, select_formal_cached_batches
import runner
from test_formal_cifar100_metrics import _CountingFormalLogitTrainer


class TaskAffineCalibratorTests(unittest.TestCase):
    def test_cached_bic_accepts_only_exact_precomputed_test_logits(self):
        from adaptive_consolidation_audit import _formal_cache_identity

        labels = torch.tensor([2, 0, 4, 1, 2, 0, 4, 1])
        inputs = torch.arange(8).float().view(-1, 1)
        logits = torch.arange(48).float().reshape(8, 6) / 13
        logits[torch.arange(8), labels] += 0.5
        cached = cache_formal_batches(((inputs[:3], labels[:3]),
                                      (inputs[3:], labels[3:])))
        tasks = {0: [2, 0], 1: [4, 1]}
        seen = (2, 0, 4, 1)
        selected = select_formal_cached_batches(cached, seen)
        audit = {'passed': True, 'test_used_for_fit': False}
        corpus = {'split': 'calibration', 'manifest_sha256': 'a' * 64}
        args = SimpleNamespace(bic_lr=0.05, bic_steps=3)
        fit_inputs, paired_inputs = [], []
        original_fit = runner.fit_final_calibrator
        original_summary = runner.summarize_paired_logits

        def fit(*values, **kwargs):
            fit_inputs.append((values[0].clone(), values[1].clone()))
            return original_fit(*values, **kwargs)

        def summarize(*values, **kwargs):
            paired_inputs.append((values[0], values[1]))
            return original_summary(*values, **kwargs)

        reference = _CountingFormalLogitTrainer(logits)
        current = _CountingFormalLogitTrainer(logits)
        reference_calibrator, current_calibrator = TaskAffineCalibrator(), TaskAffineCalibrator()
        cache_before = _formal_cache_identity(cached)
        rng_before = runner._capture_rng_state()
        with mock.patch('runner.fit_final_calibrator', side_effect=fit), mock.patch(
                'runner.summarize_paired_logits', side_effect=summarize):
            reference_record = runner._fit_and_evaluate_final_bic_cached(
                reference, cached, cached, reference_calibrator, args,
                'event_1_CIL', tasks, audit, corpus,
            )
            bundle = {'logits': logits.clone(), 'labels': labels.clone(),
                      'seen_classes': seen}
            current_record = runner._fit_and_evaluate_final_bic_cached(
                current, cached, cached, current_calibrator, args,
                'event_1_CIL', tasks, audit, corpus, precomputed_test=bundle,
            )

        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
        self.assertEqual(canonical(current_record), canonical(reference_record))
        self.assertEqual(canonical(current_calibrator.state_dict()),
                         canonical(reference_calibrator.state_dict()))
        for before, after in zip(fit_inputs[0], fit_inputs[1]):
            self.assertTrue(torch.equal(before, after))
        for before, after in zip(paired_inputs[0], paired_inputs[1]):
            self.assertTrue(torch.equal(before, after))
        self.assertIs(paired_inputs[1][0], bundle['logits'])
        self.assertIs(paired_inputs[1][1], bundle['labels'])
        self.assertEqual(reference.calls, ['collect', 'collect'])
        self.assertEqual(current.calls, ['collect'])
        self.assertEqual(current.forward_rows, reference.forward_rows[:len(selected)])
        self.assertEqual(_formal_cache_identity(cached), cache_before)
        self.assertTrue(runner._checkpoint_values_equal(
            runner._capture_rng_state(), rng_before))

        invalid = {
            'missing_key': lambda value: value.pop('labels'),
            'extra_key': lambda value: value.update(extra=True),
            'missing_rows': lambda value: value.update(
                logits=value['logits'][:-1], labels=value['labels'][:-1]),
            'extra_rows': lambda value: value.update(
                logits=torch.cat((value['logits'], value['logits'][:1])),
                labels=torch.cat((value['labels'], value['labels'][:1]))),
            'label_order': lambda value: value.update(labels=value['labels'].flip(0)),
            'wrong_classes': lambda value: value.update(seen_classes=(0, 2, 4, 1)),
            'class_list': lambda value: value.update(seen_classes=list(seen)),
            'float64': lambda value: value.update(logits=value['logits'].double()),
            'int32_labels': lambda value: value.update(labels=value['labels'].int()),
            'non_cpu_logits': lambda value: value.update(logits=value['logits'].to('meta')),
            'non_cpu_labels': lambda value: value.update(labels=value['labels'].to('meta')),
            'nonfinite': lambda value: value['logits'].__setitem__((0, 0), float('nan')),
            'requires_grad': lambda value: value['logits'].requires_grad_(True),
            'wrong_rank': lambda value: value.update(logits=value['logits'].flatten()),
            'missing_column': lambda value: value.update(logits=value['logits'][:, :4]),
        }
        for name, mutate in invalid.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(bundle)
                mutate(malformed)
                trainer = _CountingFormalLogitTrainer(logits)
                with self.assertRaisesRegex(ValueError, 'formal union logits'):
                    runner._fit_and_evaluate_final_bic_cached(
                        trainer, cached, cached, TaskAffineCalibrator(), args,
                        'event_1_CIL', tasks, audit, corpus,
                        precomputed_test=malformed,
                    )
                self.assertEqual(trainer.calls, [], 'malformed input must not collect fallback')

        for member in ('logits', 'labels', 'seen_classes'):
            with self.subTest(mutation_after_handoff=member):
                handed = copy.deepcopy(bundle)

                def mutate_after_summary(*values, **kwargs):
                    result = original_summary(*values, **kwargs)
                    if member == 'seen_classes':
                        handed[member] = tuple(reversed(seen))
                    else:
                        handed[member].view(-1)[0] += 1
                    return result

                with mock.patch('runner.summarize_paired_logits',
                                side_effect=mutate_after_summary), self.assertRaisesRegex(
                        ValueError, 'formal union logits'):
                    runner._fit_and_evaluate_final_bic_cached(
                        _CountingFormalLogitTrainer(logits), cached, cached,
                        TaskAffineCalibrator(), args, 'event_1_CIL', tasks,
                        audit, corpus, precomputed_test=handed,
                    )

    def test_final_modes_are_deterministic_and_respect_frozen_parameters(self):
        logits = torch.tensor([
            [3.0, 1.0, 5.0, 4.0], [1.0, 3.0, 4.0, 5.0],
            [3.2, 1.0, 5.1, 4.0], [1.0, 3.2, 4.0, 5.1],
            [0.0, 1.0, 5.0, 2.0], [1.0, 0.0, 2.0, 5.0],
            [0.0, 1.0, 4.8, 2.0], [1.0, 0.0, 2.0, 4.8],
        ])
        labels = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
        task_classes = {0: [0, 1], 1: [2, 3]}

        for mode in ('beta_only', 'alpha_only', 'joint_alpha_beta'):
            first, audit_a = fit_final_calibrator(
                logits, labels, task_classes, mode, lr=0.05, steps=200
            )
            second, audit_b = fit_final_calibrator(
                logits, labels, task_classes, mode, lr=0.05, steps=200
            )

            self.assertLess(audit_a['loss_after'], audit_a['loss_before'])
            self.assertEqual(audit_a, audit_b)
            self.assertEqual(first.state_dict(), second.state_dict())
            params = [first.parameters_for(task_id) for task_id in task_classes]
            if mode == 'beta_only':
                self.assertTrue(all(abs(item['alpha'] - 1.0) < 1e-6 for item in params))
            if mode == 'alpha_only':
                self.assertTrue(all(item['beta'] == 0.0 for item in params))

    def test_final_calibrator_rejects_invalid_inputs(self):
        logits = torch.zeros(2, 4)
        labels = torch.tensor([0, 9])
        with self.assertRaises(ValueError):
            fit_final_calibrator(
                logits, labels, {0: [0, 1], 1: [2, 3]}, 'beta_only', 0.05, 10
            )

    def test_positive_scale_and_within_task_argmax_are_preserved(self):
        calibrator = TaskAffineCalibrator()
        calibrator.set_task(0, [0, 1], raw_alpha=0.2, beta=-0.4)
        logits = torch.tensor([[3.0, 1.0, 0.0, 2.0]])

        calibrated = calibrator.apply(logits)

        self.assertGreater(calibrator.parameters_for(0)['alpha'], 0.0)
        self.assertTrue(logits[:, :2].argmax(1).equal(calibrated[:, :2].argmax(1)))

    def test_fit_is_deterministic_and_reduces_cross_entropy(self):
        logits = torch.tensor([
            [3.0, 1.0, 4.0, 3.0], [1.0, 3.0, 3.0, 4.0],
            [3.2, 1.0, 4.1, 3.0], [1.0, 3.2, 3.0, 4.1],
            [0.0, 1.0, 5.0, 2.0], [1.0, 0.0, 2.0, 5.0],
            [0.0, 1.0, 4.8, 2.0], [1.0, 0.0, 2.0, 4.8],
        ])
        labels = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])

        first = TaskAffineCalibrator()
        second = TaskAffineCalibrator()
        result_a = first.fit_task(logits, labels, 1, [2, 3], [0, 1, 2, 3], 0.05, 200)
        result_b = second.fit_task(logits, labels, 1, [2, 3], [0, 1, 2, 3], 0.05, 200)

        self.assertLess(result_a['loss_after'], result_a['loss_before'])
        self.assertEqual(result_a, result_b)
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_state_round_trip_preserves_logits(self):
        calibrator = TaskAffineCalibrator()
        calibrator.set_task(2, [4, 5], raw_alpha=-0.3, beta=0.7)
        restored = TaskAffineCalibrator()
        restored.load_state_dict(calibrator.state_dict())
        logits = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        self.assertTrue(torch.equal(calibrator.apply(logits), restored.apply(logits)))

    def test_paired_summary_reports_task_fractions_and_preserves_task_il(self):
        logits = torch.tensor([
            [3.0, 1.0, 5.0, 4.0],
            [1.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 5.0, 2.0],
            [1.0, 0.0, 2.0, 5.0],
        ])
        labels = torch.tensor([0, 1, 2, 3])
        calibrator = TaskAffineCalibrator()
        calibrator.set_task(1, [2, 3], raw_alpha=-2.0, beta=-2.0)

        summary = summarize_paired_logits(
            logits, labels, {0: [0, 1], 1: [2, 3]}, calibrator
        )

        self.assertLess(summary['calibrated']['task_prediction_fraction']['task_1'],
                        summary['raw']['task_prediction_fraction']['task_1'])
        self.assertEqual(summary['raw']['task_il'], summary['calibrated']['task_il'])

    def test_cached_joint_stage_reconstructs_exact_record_and_float32_state(self):
        fit_cached = getattr(runner, '_fit_and_evaluate_final_bic_cached', None)
        self.assertTrue(callable(fit_cached), 'missing cached BiC stage adapter')
        task_classes = {task: [2 * task, 2 * task + 1] for task in range(10)}
        labels = torch.arange(20, dtype=torch.long).repeat(2)
        inputs = labels.to(torch.float32).view(-1, 1)
        cache = cache_formal_batches(((inputs[:20], labels[:20]),
                                      (inputs[20:], labels[20:])))

        class CachedTrainer:
            def __init__(self):
                self.calls = []

            def collect_logits(self, batches):
                values = torch.cat([value for value, _labels in batches])
                actual = torch.cat([value for _inputs, value in batches])
                self.calls.append((values.clone(), actual.clone()))
                logits = torch.full((actual.numel(), 20), -2.0, dtype=torch.float32)
                logits[torch.arange(actual.numel()), actual] = 3.0
                return logits, actual

        trainer = CachedTrainer()
        calibrator = TaskAffineCalibrator()
        audit = {
            'passed': True, 'manifest_sha256': 'a' * 64,
            'per_class': 2, 'calibration_count': 40,
            'training_count': 60, 'overlap_count': 0,
            'test_used_for_fit': False,
        }
        record = fit_cached(
            trainer, cache, cache, calibrator,
            SimpleNamespace(bic_lr=0.05, bic_steps=2),
            'event_9_CIL', task_classes, audit,
        )

        self.assertEqual(set(record), {
            'step', 'task_id', 'fit', 'paired', 'parameters',
            'calibration_audit', 'disabled_identity_max_abs_diff',
            'task_il_max_abs_delta', 'privacy_audit',
        })
        self.assertEqual(record['task_id'], 9)
        self.assertEqual(record['fit']['mode'], 'joint_alpha_beta')
        self.assertEqual(set(record['parameters']), {str(i) for i in range(10)})
        self.assertEqual(record['calibration_audit'], audit)
        self.assertEqual(record['privacy_audit'], {
            'passed': True, 'test_used_for_fit': False,
            'raw_images_saved': False, 'party_embeddings_saved': False,
        })
        self.assertEqual(len(trainer.calls), 2)
        self.assertTrue(all(values.dtype == torch.float32
                            for values, _labels in trainer.calls))
        restored = TaskAffineCalibrator()
        restored.load_state_dict(calibrator.state_dict())
        self.assertEqual(restored.state_dict(), calibrator.state_dict())
        self.assertEqual(
            {str(task): restored.parameters_for(task) for task in range(10)},
            record['parameters'],
        )

    def test_formal_bic_cache_plan_has_exact_cifar_phase_and_count(self):
        plan = getattr(runner, '_formal_bic_cache_plan', None)
        self.assertTrue(callable(plan), 'missing formal BiC cache phase adapter')
        base = dict(
            data='cifar100', num_tasks=10, bic_enabled=1,
            bic_fit_mode='joint_each_stage', cl_method='finetune',
            lambda_validation_enabled=1,
            head_consolidation_enabled=0,
            head_consolidation_mode='full_classifier',
        )
        self.assertEqual(
            plan(SimpleNamespace(**base)),
            ('calibration', 'final_bic_calibration_post_freeze'),
        )
        internal = dict(base)
        internal.update(
            cl_method='proto_evolve', head_consolidation_enabled=1,
            head_consolidation_mode='adaptive_dual_branch',
        )
        self.assertEqual(
            plan(SimpleNamespace(**internal)),
            ('validation', 'final_validation_pre_install'),
        )
        for changed in (
                {'data': 'isolet'}, {'num_tasks': 9},
                {'bic_fit_mode': 'joint_final'},
                {'lambda_validation_enabled': 0}):
            with self.subTest(changed=changed):
                invalid = dict(base)
                invalid.update(changed)
                with self.assertRaises(ValueError):
                    plan(SimpleNamespace(**invalid))


if __name__ == '__main__':
    unittest.main()
