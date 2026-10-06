import unittest
from unittest.mock import patch

import torch

import runner
from metrics import MetricsTracker
from runner import _evaluate_cil_readouts


class HookMethod:
    def evaluate_class_il_readouts(self, dataset, task_classes):
        return {
            "class_il_pred_task": {"task_0": 0.6, "task_1": 0.4},
            "class_il_global": {"task_0": 0.2, "task_1": 0.1},
            "task_il_oracle": {"task_0": 0.9, "task_1": 0.8},
            "task_prediction": {"task_0": 0.7, "task_1": 0.5},
            "class_il_pred_task_counts": {
                "task_0": {"correct": 3, "total": 5},
                "task_1": {"correct": 2, "total": 5},
            },
        }


class ImbalancedHookMethod:
    def evaluate_class_il_readouts(self, dataset, task_classes):
        return {
            "class_il_pred_task": {"task_0": 1.0, "task_1": 0.0},
            "class_il_global": {"task_0": 1.0, "task_1": 0.0},
            "task_il_oracle": {"task_0": 1.0, "task_1": 0.0},
            "task_prediction": {"task_0": 1.0, "task_1": 0.0},
            "class_il_pred_task_counts": {
                "task_0": {"correct": 3, "total": 3},
                "task_1": {"correct": 0, "total": 1},
            },
        }

    def evaluate_class_il_readouts_cached(self, cached_batches, task_classes):
        return self.evaluate_class_il_readouts(None, task_classes)


class LegacyMethod:
    pass


class CachedHookMethod:
    def __init__(self):
        self.calls = 0

    def evaluate_class_il_readouts_cached(self, cached_batches, task_classes):
        self.calls += 1
        self.cached_batches = cached_batches
        self.task_classes = task_classes
        return HookMethod().evaluate_class_il_readouts(None, task_classes)


class RunnerFedProTIPTests(unittest.TestCase):
    def test_hook_primary_is_predicted_task_class_il(self):
        result = _evaluate_cil_readouts(
            HookMethod(), object(), object(), {0: [0, 1], 1: [2, 3]},
            [0, 1, 2, 3], "cpu",
        )
        self.assertEqual(result["primary"], {"task_0": 0.6, "task_1": 0.4})
        self.assertEqual(result["overall"], 0.5)
        self.assertEqual(result["task_il"], {"task_0": 0.9, "task_1": 0.8})
        self.assertEqual(
            set(result["companions"]), {"class_il_global", "task_prediction"}
        )

    def test_legacy_unweighted_and_formal_cached_weighted_overall_are_isolated(self):
        task_classes = {0: [0], 1: [1]}
        legacy = _evaluate_cil_readouts(
            ImbalancedHookMethod(), object(), object(), task_classes,
            [0, 1], "cpu",
        )
        cache = ((torch.zeros(4, 1), torch.tensor([0, 0, 0, 1])),)
        formal = runner._evaluate_cil_readouts_cached(
            ImbalancedHookMethod(), object(), cache, task_classes, "cpu",
        )
        self.assertEqual(legacy["overall"], 0.5)
        self.assertEqual(formal["overall"], 0.75)

    @patch("runner.evaluate_per_task_full")
    def test_legacy_method_keeps_existing_evaluator(self, evaluate):
        evaluate.return_value = (
            {"task_0": 0.3}, {"task_0": 0.4}, {"task_0": 0.8}
        )
        trainer = type(
            "Trainer", (), {"evaluate": lambda self, loader: (0.3, None, None)}
        )()
        dataset = type(
            "Dataset",
            (),
            {"get_task_loaders": lambda self, classes, shuffle_train=False: (None, object())},
        )()
        result = _evaluate_cil_readouts(
            LegacyMethod(), trainer, dataset, {0: [0, 1]}, [0, 1], "cpu"
        )
        self.assertEqual(result["primary"], {"task_0": 0.3})
        self.assertEqual(result["debiased"], {"task_0": 0.4})
        self.assertEqual(result["task_il"], {"task_0": 0.8})
        self.assertEqual(result["companions"], {})

    def test_tracker_summarizes_companions_without_renaming_primary(self):
        tracker = MetricsTracker()
        tracker.record_task_accuracies(
            "event_0_CIL",
            {"task_0": 0.5},
            0.5,
            companion_readouts={
                "class_il_global": {"task_0": 0.2},
                "task_prediction": {"task_0": 0.6},
            },
        )
        metrics = tracker.compute_cl_metrics()
        self.assertEqual(metrics["AA_final"], 0.5)
        self.assertEqual(metrics["AA_final_class_il_global"], 0.2)
        self.assertEqual(metrics["AA_final_task_prediction"], 0.6)

    def test_formal_cached_hook_preserves_fedprotip_primary_without_loader(self):
        evaluate = getattr(runner, "_evaluate_cil_readouts_cached", None)
        self.assertTrue(callable(evaluate), "missing formal cached readout adapter")
        method = CachedHookMethod()
        cache = ((
            torch.zeros(10, 1),
            torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]),
        ),)
        result = evaluate(method, object(), cache, {0: [0, 1], 1: [2, 3]}, "cpu")
        self.assertEqual(method.calls, 1)
        self.assertIs(method.cached_batches, cache)
        self.assertEqual(result["primary"], {"task_0": 0.6, "task_1": 0.4})
        self.assertEqual(result["overall"], 0.5)
        self.assertEqual(result["task_il"], {"task_0": 0.9, "task_1": 0.8})
        self.assertEqual(
            set(result["companions"]), {"class_il_global", "task_prediction"}
        )


if __name__ == "__main__":
    unittest.main()
