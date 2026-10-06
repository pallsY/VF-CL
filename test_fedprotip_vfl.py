import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from cl_methods.fedprotip_vfl import (
    FedProTIPVFLCL,
    energy_basis,
    masked_predictions,
    normalized_relevance,
    vote_tasks,
)
from metrics import cache_formal_batches
from runner import _checkpoint_values_equal


class FedProTIPPrimitiveTests(unittest.TestCase):
    def test_energy_basis_uses_smallest_positive_rank_at_threshold(self):
        centered = torch.diag(torch.tensor([3.0, 1.0]))
        basis = energy_basis(centered, 0.775)
        self.assertEqual(tuple(basis.shape), (2, 1))
        self.assertTrue(torch.allclose(basis.abs(), torch.tensor([[1.0], [0.0]])))

    def test_energy_basis_rejects_invalid_or_zero_energy_inputs(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            energy_basis(torch.ones(3), 0.775)
        with self.assertRaisesRegex(ValueError, "finite"):
            energy_basis(torch.tensor([[float("nan")]]), 0.775)
        with self.assertRaisesRegex(ValueError, "zero energy"):
            energy_basis(torch.zeros(3, 2), 0.775)

    def test_normalized_relevance_is_scale_normalized(self):
        mean = torch.zeros(2)
        basis = torch.tensor([[1.0], [0.0]])
        embeddings = torch.tensor([[2.0, 0.0], [20.0, 0.0], [0.0, 4.0]])
        score = normalized_relevance(embeddings, mean, basis)
        self.assertTrue(torch.allclose(score, torch.tensor([1.0, 1.0, 0.0])))

    def test_vote_uses_majority_then_mean_score_then_smallest_task(self):
        majority = torch.tensor([
            [[0.9, 0.1]],
            [[0.8, 0.2]],
            [[0.1, 0.9]],
        ])
        self.assertEqual(vote_tasks(majority, [3, 7]).tolist(), [3])

        mean_breaks_tie = torch.tensor([
            [[0.8, 0.7]],
            [[0.1, 0.9]],
        ])
        self.assertEqual(vote_tasks(mean_breaks_tie, [3, 7]).tolist(), [7])

        exact_tie = torch.tensor([
            [[0.9, 0.1]],
            [[0.1, 0.9]],
        ])
        self.assertEqual(vote_tasks(exact_tie, [3, 7]).tolist(), [3])

    def test_masked_predictions_never_select_outside_predicted_task(self):
        logits = torch.tensor([[0.0, 9.0, 3.0, 4.0], [8.0, 7.0, 0.0, 1.0]])
        pred_tasks = torch.tensor([0, 1])
        task_classes = {0: [0, 1], 1: [2, 3]}
        self.assertEqual(
            masked_predictions(logits, pred_tasks, task_classes).tolist(),
            [1, 3],
        )


class IdentityBottom(torch.nn.Module):
    def forward(self, value):
        return value.flatten(1)


class FakeTop(torch.nn.Module):
    def forward(self, embedding):
        return torch.cat([embedding, embedding], dim=1)


class FakeTrainer:
    def __init__(self):
        self.bottoms = [IdentityBottom(), IdentityBottom()]
        self.top_model = FakeTop()

    def _aggregate(self, embeddings):
        return sum(embeddings)


class LoaderDataset:
    def __init__(self, loader):
        self.loader = loader

    def get_task_loaders(self, classes, shuffle_train=False):
        return self.loader, self.loader


class FedProTIPMethodTests(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(
            num_parties=2,
            device="cpu",
            aggregation="sum",
            epochs_per_task=1,
            gpm_threshold=0.95,
            img_size=4,
            party_widths=None,
            data="cifar100",
        )
        self.method = FedProTIPVFLCL(FakeTrainer(), args)

    def _fit_references(self):
        self.method._update_task_reference(
            [torch.tensor([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
             torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])],
            task_id=0,
            task_classes=[0, 1],
        )
        self.method._update_task_reference(
            [torch.tensor([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]),
             torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])],
            task_id=1,
            task_classes=[2, 3],
        )

    def test_threshold_is_frozen_and_reference_state_has_no_examples(self):
        self.assertEqual(self.method.threshold, 0.775)
        self.assertEqual(self.method.tip_threshold, 0.775)
        self._fit_references()
        state = self.method.get_state()
        self.assertEqual(
            set(state),
            {"feature_lists", "head_basis", "threshold", "tip_threshold",
             "max_batches", "task_means", "task_bases", "task_classes"},
        )
        self.assertNotIn("embeddings", repr(state).lower())
        self.assertNotIn("images", repr(state).lower())

    def test_prediction_does_not_accept_or_read_labels(self):
        self._fit_references()
        embeddings = [torch.tensor([[5.0, 0.0], [0.0, 5.0]])] * 2
        self.assertEqual(self.method.predict_tasks(embeddings).tolist(), [0, 1])
        with self.assertRaises(TypeError):
            self.method.predict_tasks(embeddings, torch.tensor([1, 0]))

    def test_state_round_trip_reproduces_predictions_exactly(self):
        self._fit_references()
        embeddings = [torch.tensor([[5.0, 0.0], [0.0, 5.0]])] * 2
        expected = self.method.predict_tasks(embeddings)
        restored = FedProTIPVFLCL(self.method.trainer, self.method.args)
        restored.load_state(self.method.get_state())
        self.assertTrue(torch.equal(restored.predict_tasks(embeddings), expected))
        self.assertEqual(restored.task_classes, self.method.task_classes)

    def test_instance_options_drive_threshold_and_after_task_batch_limit(self):
        args = SimpleNamespace(
            **vars(self.method.args),
            fedprotip_tip_threshold=0.8,
            fedprotip_max_batches=3,
        )
        method = FedProTIPVFLCL(self.method.trainer, args)
        self.assertEqual((method.threshold, method.tip_threshold), (0.8, 0.8))
        self.assertEqual(method.max_batches, 3)
        method._pending_task_classes[0] = [0, 1]
        embeddings = [
            torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        ]
        with mock.patch(
                'cl_methods.fedprotip_vfl.GPMCL.after_task'), mock.patch.object(
                    method, '_collect_party_embeddings',
                    return_value=(embeddings, torch.tensor([0, 1]))) as collect:
            method.after_task([], 0)
        collect.assert_called_once_with([], max_batches=3)

    def test_state_strictly_binds_instance_options(self):
        state = self.method.get_state()
        self.assertEqual(state['tip_threshold'], 0.775)
        self.assertEqual(state['max_batches'], 20)
        for field, changed in (
                ('tip_threshold', 0.8), ('max_batches', 21)):
            if field == 'tip_threshold':
                missing = dict(state)
                del missing[field]
                restored = FedProTIPVFLCL(
                    self.method.trainer, self.method.args)
                with self.assertRaisesRegex(ValueError, field):
                    restored.load_state(missing)
            with self.subTest(field=field, case='changed'):
                mismatched = dict(state)
                mismatched[field] = changed
                restored = FedProTIPVFLCL(
                    self.method.trainer, self.method.args)
                with self.assertRaisesRegex(ValueError, field):
                    restored.load_state(mismatched)

    def test_legacy_missing_max_batches_is_nonformal_only_and_new_state_is_strict(self):
        self._fit_references()
        state = self.method.get_state()
        legacy_state = dict(state)
        del legacy_state['max_batches']

        legacy = FedProTIPVFLCL(self.method.trainer, self.method.args)
        legacy.load_state(legacy_state)
        self.assertEqual(legacy.max_batches, 20)
        self.assertEqual(legacy.get_state()['max_batches'], 20)

        formal_args = SimpleNamespace(
            **vars(self.method.args), formal_deferred_evaluation=True,
        )
        with self.assertRaisesRegex(ValueError, 'max_batches'):
            FedProTIPVFLCL(self.method.trainer, formal_args).load_state(
                legacy_state
            )

        wrong = dict(state, max_batches=21)
        for args in (self.method.args, formal_args):
            with self.subTest(formal=getattr(
                    args, 'formal_deferred_evaluation', False)):
                with self.assertRaisesRegex(ValueError, 'max_batches'):
                    FedProTIPVFLCL(self.method.trainer, args).load_state(wrong)

        restored = FedProTIPVFLCL(self.method.trainer, formal_args)
        restored.load_state(state)
        self.assertTrue(_checkpoint_values_equal(restored.get_state(), state))

    def test_duplicate_classes_are_rejected(self):
        self._fit_references()
        with self.assertRaisesRegex(ValueError, "reuses an existing class"):
            self.method._update_task_reference(
                [torch.tensor([[1.0, 0.0], [2.0, 0.0]])] * 2,
                task_id=2,
                task_classes=[1, 4],
            )

    def test_readout_reports_predicted_global_oracle_and_task_accuracy(self):
        self._fit_references()
        x = torch.tensor([
            [[[4.0, 0.0, 4.0, 0.0]]],
            [[[0.0, 4.0, 0.0, 4.0]]],
        ])
        y = torch.tensor([0, 3])
        loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)
        result = self.method.evaluate_class_il_readouts(
            LoaderDataset(loader), {0: [0, 1], 1: [2, 3]}
        )
        self.assertEqual(
            set(result),
            {"class_il_pred_task", "class_il_global", "task_il_oracle",
             "task_prediction", "class_il_pred_task_counts"},
        )
        self.assertEqual(set(result["task_prediction"]), {"task_0", "task_1"})
        counts = result["class_il_pred_task_counts"]
        self.assertEqual(set(counts), {"task_0", "task_1"})
        for evidence in counts.values():
            self.assertEqual(set(evidence), {"correct", "total"})
            self.assertIs(type(evidence["correct"]), int)
            self.assertIs(type(evidence["total"]), int)
            self.assertGreater(evidence["total"], 0)
            self.assertGreaterEqual(evidence["correct"], 0)
            self.assertLessEqual(evidence["correct"], evidence["total"])

    def test_readout_is_invariant_to_loader_batch_boundaries(self):
        self._fit_references()
        x = torch.tensor([
            [[[4.0, 0.0, 4.0, 0.0]]],
            [[[0.0, 4.0, 0.0, 4.0]]],
        ])
        y = torch.tensor([0, 3])
        first = self.method.evaluate_class_il_readouts(
            LoaderDataset(DataLoader(TensorDataset(x, y), batch_size=1)),
            {0: [0, 1], 1: [2, 3]},
        )
        second = self.method.evaluate_class_il_readouts(
            LoaderDataset(DataLoader(TensorDataset(x, y), batch_size=2)),
            {0: [0, 1], 1: [2, 3]},
        )
        self.assertEqual(first, second)

    def test_cached_readout_is_identical_and_never_opens_a_loader(self):
        self._fit_references()
        x = torch.tensor([
            [[[4.0, 0.0, 4.0, 0.0]]],
            [[[0.0, 4.0, 0.0, 4.0]]],
        ])
        y = torch.tensor([0, 3])
        loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)

        class FilteringDataset:
            def get_task_loaders(self, classes, shuffle_train=False):
                mask = torch.tensor([int(label) in classes for label in y])
                selected = DataLoader(
                    TensorDataset(x[mask], y[mask]), batch_size=1,
                    shuffle=False,
                )
                return selected, selected

        expected = self.method.evaluate_class_il_readouts(
            FilteringDataset(), {0: [0, 1], 1: [2, 3]}
        )
        cache = cache_formal_batches(loader)

        class ClosedDataset:
            def get_task_loaders(self, *_args, **_kwargs):
                raise AssertionError("formal cached readout reopened a loader")

        self.assertTrue(
            callable(getattr(self.method, "evaluate_class_il_readouts_cached", None)),
            "FedProTIP needs a formal cached readout hook",
        )
        actual = self.method.evaluate_class_il_readouts_cached(
            cache, {0: [0, 1], 1: [2, 3]}
        )
        self.assertEqual(actual, expected)
        self.assertFalse(hasattr(actual, "dataset"))
        ClosedDataset()  # documents that no dataset object is passed to the hook

    def test_formal_evaluation_state_is_exact_snapshot_safe_and_strict(self):
        self._fit_references()
        getter = getattr(self.method, "get_formal_evaluation_state", None)
        self.assertTrue(callable(getter), "missing formal evaluation state adapter")
        state = getter()
        self.assertEqual(set(state), {
            "task_means", "task_bases", "task_classes",
            "tip_threshold", "max_batches",
        })
        self.assertNotIn("feature_lists", state)
        self.assertNotIn("head_basis", state)
        self.assertNotIn("dataset", repr(state).lower())
        self.assertNotIn("image", repr(state).lower())

        restored = FedProTIPVFLCL(self.method.trainer, self.method.args)
        loader = getattr(restored, "load_formal_evaluation_state", None)
        self.assertTrue(callable(loader), "missing strict formal state loader")
        loader(state)
        embeddings = [torch.tensor([[5.0, 0.0], [0.0, 5.0]])] * 2
        self.assertTrue(torch.equal(
            restored.predict_tasks(embeddings), self.method.predict_tasks(embeddings)
        ))
        for field in state:
            with self.subTest(field=field):
                tampered = dict(state)
                del tampered[field]
                with self.assertRaises((TypeError, ValueError)):
                    FedProTIPVFLCL(
                        self.method.trainer, self.method.args
                    ).load_formal_evaluation_state(tampered)

        tampered = self.method.get_formal_evaluation_state()
        tampered["task_means"][0][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            FedProTIPVFLCL(
                self.method.trainer, self.method.args
            ).load_formal_evaluation_state(tampered)


class FedProTIPRegistryTests(unittest.TestCase):
    def test_registry_builds_frozen_fedprotip_vfl(self):
        from cl_methods import get_cl_method

        args = SimpleNamespace(
            num_parties=2,
            device="cpu",
            aggregation="sum",
            epochs_per_task=1,
            gpm_threshold=0.95,
            img_size=4,
            party_widths=None,
            data="cifar100",
        )
        method = get_cl_method("fedprotip_vfl", FakeTrainer(), args)
        self.assertIsInstance(method, FedProTIPVFLCL)
        self.assertEqual(method.threshold, 0.775)


if __name__ == "__main__":
    unittest.main()
