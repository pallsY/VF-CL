import copy
import unittest

import torch

from formal_cifar100_metrics import (
    aggregate_rows,
    calibration_history,
    calibration_is_legal,
    compute_formal_metrics,
    raw_history,
)
from metrics import (
    cache_formal_batches,
    evaluate_per_task_full,
    evaluate_per_task_full_cached,
    select_formal_cached_batches,
)
from runner import _evaluate_cil_readouts, _evaluate_cil_readouts_cached
import runner


class _CountingFormalLogitTrainer:
    """Fixed full-head forward; retain original batch and row identities."""
    def __init__(self, logits):
        self.logits = logits
        self.calls = []
        self.forward_rows = []

    def _forward(self, batches, kind):
        self.calls.append(kind)
        rows, labels = [], []
        for inputs, targets in batches:
            indices = inputs[:, 0].long()
            self.forward_rows.append((kind, tuple(indices.tolist())))
            rows.append(self.logits[indices])
            labels.append(targets)
        return torch.cat(rows), torch.cat(labels)

    def evaluate(self, batches):
        logits, labels = self._forward(batches, 'evaluate')
        correct = int(logits.argmax(1).eq(labels).sum())
        return correct / labels.numel(), torch.softmax(logits, 1), labels

    def collect_logits(self, batches):
        return self._forward(batches, 'collect')


def _legacy_formal_readouts(method, trainer, cached, task_classes, device):
    """7cba0dc non-FedProTIP graph: per-task evaluate, then union evaluate."""
    effective = [int(value) for classes in task_classes.values()
                 for value in classes]
    selected = select_formal_cached_batches(cached, effective)
    observed = [label for _, labels in selected for label in labels.tolist()]
    if (any(type(label) is not int for label in observed)
            or set(observed) != set(effective)):
        raise ValueError('formal cache labels must exactly match the evaluated classes')
    raw, debiased, task_il = evaluate_per_task_full_cached(
        trainer, cached, task_classes, device
    )
    overall, _, _ = trainer.evaluate(selected)
    return {'primary': raw, 'debiased': debiased, 'task_il': task_il,
            'companions': {}, 'overall': round(float(overall), 4)}


class _CachedFormulaTrainer:
    def evaluate(self, loader):
        probabilities = torch.cat([batch_x for batch_x, _ in loader])
        labels = torch.cat([batch_y for _, batch_y in loader])
        accuracy = float((probabilities.argmax(1) == labels).float().mean())
        return accuracy, probabilities, labels


class _CachedFormulaDataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.test_accesses = []

    def get_task_loaders(self, classes, shuffle_train=False):
        self.test_accesses.append(tuple(classes))
        mask = torch.zeros_like(self.labels, dtype=torch.bool)
        for class_id in classes:
            mask |= self.labels == class_id
        return (), ((self.features[mask], self.labels[mask]),)


class _FedCachedFormulaMethod:
    def __init__(self, *, primary=None, evidence=None, include_evidence=True):
        self.primary = (
            primary if primary is not None
            else {'task_0': 1.0, 'task_1': 0.0}
        )
        self.evidence = evidence if evidence is not None else {
            'task_0': {'correct': 3, 'total': 3},
            'task_1': {'correct': 0, 'total': 1},
        }
        self.include_evidence = include_evidence

    def evaluate_class_il_readouts_cached(self, cached_batches, task_classes):
        result = {
            'class_il_pred_task': self.primary,
            'class_il_global': {'task_0': 0.5, 'task_1': 0.5},
            'task_il_oracle': {'task_0': 1.0, 'task_1': 1.0},
            'task_prediction': {'task_0': 1.0, 'task_1': 0.0},
        }
        if self.include_evidence:
            result['class_il_pred_task_counts'] = self.evidence
        return result


class FormalCifar100MetricsTests(unittest.TestCase):
    def test_formal_union_logits_reuse_is_exact_and_avoids_one_union_forward(self):
        from adaptive_consolidation_audit import _formal_cache_identity

        logits = torch.tensor([
            [0., 1., 4., 0., 2., 5.], [0., 0., 0., 0., 0., 6.],
            [4., 2., 0., 0., 1., 0.], [0., 3., 1., 0., 2., 0.],
            [0., 1., 3., 0., 2., 0.], [0., 1., 0., 0., 4., 0.],
            [4., 3., 1., 0., 2., 0.],
        ])
        labels = torch.tensor([2, 5, 0, 1, 2, 4, 1])
        inputs = torch.arange(7).float().view(-1, 1)
        cached = cache_formal_batches(((inputs[:3], labels[:3]),
                                      (inputs[3:4], labels[3:4]),
                                      (inputs[4:], labels[4:])))
        tasks = {0: [2, 0], 1: [4, 1]}
        seen = (2, 0, 4, 1)
        selected = select_formal_cached_batches(cached, seen)
        cache_before = _formal_cache_identity(cached)
        rng_before = runner._capture_rng_state()
        reference = _CountingFormalLogitTrainer(logits)
        reference_readouts = _legacy_formal_readouts(
            object(), reference, cached, tasks, 'cpu'
        )
        reference_test_logits, reference_test_labels = reference.collect_logits(selected)
        current = _CountingFormalLogitTrainer(logits)
        readouts = _evaluate_cil_readouts_cached(
            object(), current, cached, tasks, 'cpu', collect_union_logits=True
        )
        bundle = readouts.pop('formal_union_logits')

        self.assertEqual(readouts, reference_readouts)
        self.assertEqual(set(bundle), {'logits', 'labels', 'seen_classes'})
        self.assertTrue(torch.equal(bundle['logits'], reference_test_logits))
        self.assertTrue(torch.equal(bundle['labels'], reference_test_labels))
        self.assertTrue(torch.equal(bundle['logits'].argmax(1),
                                    reference_test_logits.argmax(1)))
        self.assertEqual(bundle['seen_classes'], seen)
        self.assertEqual(bundle['logits'].dtype, torch.float32)
        self.assertEqual(bundle['labels'].dtype, torch.long)
        for value in (bundle['logits'], bundle['labels']):
            self.assertEqual(value.device.type, 'cpu')
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        per_task_calls = len(reference.forward_rows) - 2 * len(selected)
        self.assertEqual(current.forward_rows[:per_task_calls],
                         reference.forward_rows[:per_task_calls])
        reference_union_rows = sum(len(rows) for _, rows in
                                   reference.forward_rows[per_task_calls:])
        new_union_rows = sum(len(rows) for _, rows in
                             current.forward_rows[per_task_calls:])
        self.assertEqual(reference_union_rows, 12)
        self.assertEqual(new_union_rows, 6)
        self.assertEqual(reference_union_rows - len(reference_test_labels), new_union_rows)
        self.assertEqual([rows for _, rows in current.forward_rows[per_task_calls:]],
                         [rows for _, rows in reference.forward_rows[-len(selected):]])
        self.assertEqual(_formal_cache_identity(cached), cache_before)
        self.assertTrue(runner._checkpoint_values_equal(
            runner._capture_rng_state(), rng_before))

        for name, changed in (
                ('missing', reference_test_labels[:-1]),
                ('extra', torch.cat((reference_test_labels, torch.tensor([2])))),
                ('reordered', reference_test_labels.flip(0))):
            with self.subTest(name=name):
                class InvalidLabels:
                    def collect_logits(self, _selected):
                        return reference_test_logits, changed
                with self.assertRaisesRegex(ValueError, 'formal union logits'):
                    runner._formal_union_logits(InvalidLabels(), selected, seen)

    def _readout_results(self, probabilities, labels, tasks):
        trainer = _CachedFormulaTrainer()
        dataset = _CachedFormulaDataset(probabilities, labels)
        loader_result = _evaluate_cil_readouts(
            object(), trainer, dataset, tasks,
            [class_id for classes in tasks.values() for class_id in classes],
            'cpu',
        )
        cached_result = _evaluate_cil_readouts_cached(
            object(), trainer,
            cache_formal_batches(((probabilities, labels),)), tasks, 'cpu',
        )
        return loader_result, cached_result

    def test_cached_formula_matches_loader_formula_without_reopen_or_mutation(self):
        probabilities = torch.tensor([
            [0.8, 0.1, 0.1, 0.0],
            [0.4, 0.5, 0.1, 0.0],
            [0.2, 0.3, 0.4, 0.1],
            [0.1, 0.2, 0.1, 0.6],
            [0.3, 0.4, 0.2, 0.1],
            [0.1, 0.2, 0.6, 0.1],
        ], requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 2])
        source = [(probabilities[:2], labels[:2]),
                  (probabilities[2:], labels[2:])]
        source_before = tuple(
            (batch_x.detach().clone(), batch_y.clone())
            for batch_x, batch_y in source
        )
        cache = cache_formal_batches(source)

        self.assertIsInstance(cache, tuple)
        self.assertTrue(all(isinstance(batch, tuple) for batch in cache))
        for batch_x, batch_y in cache:
            self.assertEqual(batch_x.device.type, 'cpu')
            self.assertEqual(batch_y.device.type, 'cpu')
            self.assertFalse(batch_x.requires_grad)
            self.assertFalse(batch_y.requires_grad)
            self.assertIsNone(batch_x.grad_fn)
            self.assertIsNone(batch_y.grad_fn)
        self.assertNotEqual(cache[0][0].data_ptr(), probabilities.data_ptr())

        tasks = {0: [0, 1], 1: [2, 3]}
        dataset = _CachedFormulaDataset(probabilities.detach(), labels)
        loader_result = evaluate_per_task_full(
            _CachedFormulaTrainer(), dataset, tasks, 'cpu'
        )
        accesses_after_loader = list(dataset.test_accesses)
        cache_before = copy.deepcopy(cache)
        cached_result = evaluate_per_task_full_cached(
            _CachedFormulaTrainer(), cache, tasks, 'cpu'
        )

        self.assertEqual(cached_result, loader_result)
        self.assertEqual(dataset.test_accesses, accesses_after_loader)
        for before, after in zip(cache_before, cache):
            self.assertTrue(torch.equal(before[0], after[0]))
            self.assertTrue(torch.equal(before[1], after[1]))
        for before, after in zip(source_before, source):
            self.assertTrue(torch.equal(before[0], after[0]))
            self.assertTrue(torch.equal(before[1], after[1]))

    def test_cached_overall_is_sample_weighted_and_matches_loader_path(self):
        probabilities = torch.tensor([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ])
        labels = torch.tensor([0, 0, 0, 1])
        loader_result, cached_result = self._readout_results(
            probabilities, labels, {0: [0], 1: [1]}
        )

        self.assertEqual(cached_result['primary'], {
            'task_0': 1.0, 'task_1': 0.0,
        })
        self.assertEqual(loader_result['overall'], 0.75)
        self.assertEqual(cached_result['overall'], 0.75)
        self.assertEqual(cached_result['overall'], loader_result['overall'])

    def test_cached_overall_handles_isolet_style_unequal_task_class_counts(self):
        probabilities = torch.tensor([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.8, 0.1, 0.0],
            [0.9, 0.0, 0.0, 0.1, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.1],
        ])
        labels = torch.tensor([0, 1, 2, 3, 4])
        loader_result, cached_result = self._readout_results(
            probabilities, labels, {0: [0, 1, 2], 1: [3, 4]}
        )

        self.assertEqual(loader_result['overall'], 0.6)
        self.assertEqual(cached_result['overall'], 0.6)
        self.assertEqual(cached_result['overall'], loader_result['overall'])

    def test_cached_overall_balanced_case_is_unchanged(self):
        probabilities = torch.tensor([
            [0.9, 0.1],
            [0.4, 0.6],
            [0.4, 0.6],
            [0.7, 0.3],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        loader_result, cached_result = self._readout_results(
            probabilities, labels, {0: [0], 1: [1]}
        )

        self.assertEqual(loader_result['overall'], 0.5)
        self.assertEqual(cached_result['overall'], 0.5)
        self.assertEqual(cached_result['overall'], loader_result['overall'])

    def test_cached_overall_rejects_empty_missing_and_unexpected_classes(self):
        trainer = _CachedFormulaTrainer()
        cases = (
            ('empty', (), {0: [0, 1]}),
            ('missing', (
                (torch.tensor([[0.9, 0.1]]), torch.tensor([0])),
            ), {0: [0, 1]}),
            ('unexpected', ((
                torch.tensor([[0.9, 0.1, 0.0],
                              [0.1, 0.8, 0.1]]),
                torch.tensor([0, 1]),
            ),), {0: [0, 2]}),
        )
        for name, cache, tasks in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'cache|label|class'):
                _evaluate_cil_readouts_cached(
                    object(), trainer, cache, tasks, 'cpu'
                )

    def test_fedprotip_cached_overall_is_sample_weighted_with_future_classes(self):
        cache = ((
            torch.zeros(5, 2),
            torch.tensor([0, 0, 0, 1, 2]),
        ),)

        result = _evaluate_cil_readouts_cached(
            _FedCachedFormulaMethod(), object(), cache,
            {0: [0], 1: [1]}, 'cpu',
        )

        self.assertEqual(result['primary'], {
            'task_0': 1.0, 'task_1': 0.0,
        })
        self.assertEqual(result['overall'], 0.75)

    def test_fedprotip_cached_overall_rejects_missing_requested_class(self):
        cache = ((
            torch.zeros(3, 2),
            torch.tensor([0, 0, 2]),
        ),)

        with self.assertRaisesRegex(ValueError, 'cache|label|class'):
            _evaluate_cil_readouts_cached(
                _FedCachedFormulaMethod(), object(), cache,
                {0: [0], 1: [1]}, 'cpu',
            )

    def test_fedprotip_cached_overall_uses_exact_integer_evidence(self):
        cache = ((
            torch.zeros(10, 2),
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2]),
        ),)
        method = _FedCachedFormulaMethod(
            primary={'task_0': 0.6667, 'task_1': 0.1667},
            evidence={
                'task_0': {'correct': 2, 'total': 3},
                'task_1': {'correct': 1, 'total': 6},
            },
        )

        try:
            result = _evaluate_cil_readouts_cached(
                method, object(), cache, {0: [0], 1: [1]}, 'cpu'
            )
        except ValueError as error:
            self.fail(f'exact FedProTIP evidence was rejected: {error}')

        self.assertEqual(result['primary'], {
            'task_0': 0.6667, 'task_1': 0.1667,
        })
        self.assertEqual(result['overall'], 0.3333)

    def test_fedprotip_cached_overall_rejects_invalid_exact_evidence(self):
        cache = ((
            torch.zeros(10, 2),
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 2]),
        ),)
        valid = {
            'task_0': {'correct': 2, 'total': 3},
            'task_1': {'correct': 1, 'total': 6},
        }
        cases = (
            ('missing_evidence', None, False),
            ('empty_evidence', {}, True),
            ('missing_task', {'task_0': valid['task_0']}, True),
            ('extra_task', {**valid, 'task_2': {'correct': 0, 'total': 1}}, True),
            ('missing_member', {**valid, 'task_0': {'correct': 2}}, True),
            ('extra_member', {
                **valid, 'task_0': {'correct': 2, 'total': 3, 'extra': 0},
            }, True),
            ('bool_correct', {
                **valid, 'task_0': {'correct': True, 'total': 3},
            }, True),
            ('float_total', {
                **valid, 'task_0': {'correct': 2, 'total': 3.0},
            }, True),
            ('negative_correct', {
                **valid, 'task_0': {'correct': -1, 'total': 3},
            }, True),
            ('overflow_correct', {
                **valid, 'task_0': {'correct': 4, 'total': 3},
            }, True),
            ('zero_total', {
                **valid, 'task_0': {'correct': 0, 'total': 0},
            }, True),
            ('count_mismatch', {
                **valid, 'task_1': {'correct': 1, 'total': 5},
            }, True),
        )
        for name, evidence, include_evidence in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'FedProTIP|evidence|count|readout'):
                _evaluate_cil_readouts_cached(
                    _FedCachedFormulaMethod(
                        primary={'task_0': 0.6667, 'task_1': 0.1667},
                        evidence=evidence,
                        include_evidence=include_evidence,
                    ),
                    object(), cache, {0: [0], 1: [1]}, 'cpu',
                )
        with self.assertRaisesRegex(
                ValueError, 'FedProTIP|evidence|count|readout'):
            _evaluate_cil_readouts_cached(
                _FedCachedFormulaMethod(
                    primary={'task_0': 0.5, 'task_1': 0.1667},
                    evidence=valid,
                ),
                object(), cache, {0: [0], 1: [1]}, 'cpu',
            )

    def test_metrics_use_final_minus_diagonal_bwt(self):
        history = [
            {
                "step": "event_0_CIL",
                "per_task_accs": {"task_0": 0.8},
                "per_task_accs_taskil": {"task_0": 0.9},
            },
            {
                "step": "event_1_CIL",
                "per_task_accs": {"task_0": 0.6, "task_1": 0.7},
                "per_task_accs_taskil": {"task_0": 0.85, "task_1": 0.8},
            },
            {
                "step": "event_2_CIL",
                "per_task_accs": {
                    "task_0": 0.5,
                    "task_1": 0.6,
                    "task_2": 0.9,
                },
                "per_task_accs_taskil": {
                    "task_0": 0.9,
                    "task_1": 0.8,
                    "task_2": 0.7,
                },
            },
        ]

        metrics = compute_formal_metrics(history, expected_tasks=3)

        self.assertAlmostEqual(metrics["aa_final_cil"], 2.0 / 3.0)
        self.assertAlmostEqual(
            metrics["aa_avg_cil"], (0.8 + 0.65 + 2.0 / 3.0) / 3.0
        )
        self.assertAlmostEqual(metrics["bwt_cil"], -0.2)
        self.assertAlmostEqual(metrics["task_il_final"], 0.8)

    def test_fedprotip_uses_global_class_il_companion(self):
        results = {
            "task_acc_history": [
                {
                    "step": "event_0_CIL",
                    "per_task_accs": {"task_0": 0.2},
                    "per_task_accs_taskil": {"task_0": 0.8},
                    "companion_readouts": {
                        "class_il_global": {"task_0": 0.7}
                    },
                }
            ]
        }

        history = raw_history(results, {"cl_method": "fedprotip_vfl"})

        self.assertEqual(history[0]["per_task_accs"], {"task_0": 0.7})
        self.assertEqual(history[0]["per_task_accs_taskil"], {"task_0": 0.8})

    def test_legal_calibration_history_is_reusable_by_same_metric_code(self):
        event = {
            "step": "event_0_CIL",
            "paired": {
                "calibrated": {
                    "per_task_accuracy": {"task_0": 0.75},
                    "task_il": {"task_0": 0.8},
                }
            },
            "calibration_audit": {"passed": True, "test_used_for_fit": False},
            "privacy_audit": {
                "passed": True,
                "test_used_for_fit": False,
                "raw_images_saved": False,
                "party_embeddings_saved": False,
            },
        }

        self.assertTrue(calibration_is_legal([event]))
        metrics = compute_formal_metrics(
            calibration_history([event]), expected_tasks=1
        )
        self.assertEqual(metrics["aa_final_cil"], 0.75)
        self.assertEqual(metrics["task_il_final"], 0.8)

        event["privacy_audit"]["test_used_for_fit"] = True
        self.assertFalse(calibration_is_legal([event]))

    def test_aggregate_marks_missing_seed_44(self):
        rows = [
            {
                "method": "ER",
                "seed": 42,
                "aa_final_cil": 0.30,
                "aa_avg_cil": 0.40,
                "bwt_cil": -0.10,
                "task_il_final": 0.70,
                "calibration_legal": False,
            },
            {
                "method": "ER",
                "seed": 43,
                "aa_final_cil": 0.34,
                "aa_avg_cil": 0.42,
                "bwt_cil": -0.08,
                "task_il_final": 0.72,
                "calibration_legal": False,
            },
        ]

        summary = aggregate_rows(rows, expected_seeds=(42, 43, 44))[0]

        self.assertEqual(summary["n_seeds"], 2)
        self.assertEqual(summary["missing_seeds"], "44")
        self.assertFalse(summary["complete"])
        self.assertAlmostEqual(summary["aa_final_cil_mean"], 0.32)
        self.assertAlmostEqual(summary["aa_final_cil_std"], 0.02)


if __name__ == "__main__":
    unittest.main()
