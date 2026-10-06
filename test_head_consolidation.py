import unittest

import torch

import head_consolidation as head
from head_consolidation import (
    balanced_embedding_replay_batch, balanced_prototype_batch,
    consolidate_classifier, consolidate_task_class_bias,
)
from cl_methods.proto_evolve import head_consolidation_due, herding_indices
from models import TopModel


class FrozenHeadStateTests(unittest.TestCase):
    def test_hash_is_order_and_stride_independent_but_schema_sensitive(self):
        hash_top_state = getattr(head, 'hash_top_state', None)
        self.assertIsNotNone(hash_top_state, 'top-state hash API is missing')
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        first = {'z': weight.t(), 'a': torch.tensor([1, 2], dtype=torch.int16)}
        second = {
            'a': torch.tensor([1, 2], dtype=torch.int16),
            'z': weight.t().contiguous(),
        }

        self.assertEqual(hash_top_state(first), hash_top_state(second))
        self.assertNotEqual(
            hash_top_state({'value': torch.tensor([1, 2], dtype=torch.int16)}),
            hash_top_state({'value': torch.tensor([1, 2], dtype=torch.int32)}),
        )
        self.assertNotEqual(
            hash_top_state({'value': torch.arange(4, dtype=torch.uint8)}),
            hash_top_state({'value': torch.arange(4, dtype=torch.uint8).reshape(2, 2)}),
        )

    def test_freeze_state_detaches_clones_and_locks_the_mapping(self):
        freeze_state = getattr(head, 'freeze_state', None)
        self.assertIsNotNone(freeze_state, 'state freezing API is missing')
        source = torch.arange(
            6, dtype=torch.float32, requires_grad=True
        ).reshape(2, 3).t()

        frozen = freeze_state({'weight': source})

        self.assertEqual(list(frozen), ['weight'])
        self.assertEqual(frozen['weight'].device.type, 'cpu')
        self.assertTrue(frozen['weight'].is_contiguous())
        self.assertFalse(frozen['weight'].requires_grad)
        self.assertIsNone(frozen['weight'].grad_fn)
        self.assertNotEqual(frozen['weight'].data_ptr(), source.data_ptr())
        expected = frozen['weight'].clone()
        expected_hash = head.hash_top_state(frozen)
        with torch.no_grad():
            source.add_(10.0)
        torch.testing.assert_close(frozen['weight'], expected)
        frozen['weight'].zero_()
        torch.testing.assert_close(frozen['weight'], expected)
        self.assertEqual(head.hash_top_state(frozen), expected_hash)
        with self.assertRaises(TypeError):
            frozen['other'] = torch.tensor(1.0)


class HeadConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.prototypes = {
            1: {'mean': torch.tensor([-2.0, 0.0]), 'std': torch.tensor([0.1, 0.1])},
            3: {'mean': torch.tensor([2.0, 0.0]), 'std': torch.tensor([0.1, 0.1])},
        }

    def test_batch_is_balanced_deterministic_and_includes_means(self):
        first_x, first_y, classes = balanced_prototype_batch(
            self.prototypes, 4, 123, 'cpu'
        )
        second_x, second_y, _ = balanced_prototype_batch(
            self.prototypes, 4, 123, 'cpu'
        )
        self.assertEqual(classes, [1, 3])
        self.assertEqual(first_y.tolist(), [1, 1, 1, 1, 3, 3, 3, 3])
        torch.testing.assert_close(first_x, second_x)
        torch.testing.assert_close(first_y, second_y)
        torch.testing.assert_close(first_x[0], self.prototypes[1]['mean'])
        torch.testing.assert_close(first_x[4], self.prototypes[3]['mean'])

    def test_final_schedule_fires_once_while_every_fires_each_task(self):
        self.assertFalse(head_consolidation_due('final', 0, 3))
        self.assertFalse(head_consolidation_due('final', 1, 3))
        self.assertTrue(head_consolidation_due('final', 2, 3))
        self.assertTrue(all(head_consolidation_due('every', task, 3)
                            for task in range(3)))
        with self.assertRaises(ValueError):
            head_consolidation_due('unknown', 0, 3)

    def test_herding_is_deterministic_unique_and_matches_symmetric_mean(self):
        embeddings = torch.tensor([
            [-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        ])
        first = herding_indices(embeddings, 2)
        second = herding_indices(embeddings, 2)
        torch.testing.assert_close(first, second)
        self.assertEqual(len(set(first.tolist())), 2)
        selected_mean = torch.nn.functional.normalize(
            embeddings[first], dim=1
        ).mean(dim=0)
        target_mean = torch.nn.functional.normalize(
            embeddings, dim=1
        ).mean(dim=0)
        self.assertLess(float((selected_mean - target_mean).norm()), 0.25)

    def test_herding_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            herding_indices(torch.empty(0, 3), 2)
        with self.assertRaises(ValueError):
            herding_indices(torch.ones(2, 3), 0)

    def test_embedding_replay_is_balanced_and_never_pads_classes(self):
        replay = {
            1: torch.arange(12, dtype=torch.float32).reshape(6, 2),
            3: torch.arange(6, dtype=torch.float32).reshape(3, 2),
        }
        rows, labels, classes = balanced_embedding_replay_batch(replay, 4, 'cpu')
        self.assertEqual(classes, [1, 3])
        self.assertEqual(labels.tolist(), [1, 1, 1, 1, 3, 3, 3])
        torch.testing.assert_close(rows[:4], replay[1][:4])
        torch.testing.assert_close(rows[4:], replay[3])

    def test_fit_improves_ce_and_only_changes_seen_classifier_rows(self):
        torch.manual_seed(4)
        model = TopModel(2, 5, cosine=False)
        unseen_weight = model.classifier.weight[[0, 2, 4]].detach().clone()
        unseen_bias = model.classifier.bias[[0, 2, 4]].detach().clone()
        audit = consolidate_classifier(
            model,
            self.prototypes,
            regularization=0.01,
            steps=100,
            lr=0.03,
            samples_per_class=8,
            seed=99,
            device='cpu',
        )
        self.assertLess(audit['cross_entropy_after'], audit['cross_entropy_before'])
        self.assertEqual(audit['persistent_raw_example_count'], 0)
        self.assertFalse(audit['validation_used'])
        self.assertFalse(audit['test_used'])
        torch.testing.assert_close(
            model.classifier.weight[[0, 2, 4]], unseen_weight
        )
        torch.testing.assert_close(
            model.classifier.bias[[0, 2, 4]], unseen_bias
        )

    def test_fit_audits_raw_replay_without_persistent_embeddings(self):
        torch.manual_seed(9)
        model = TopModel(2, 5, cosine=False)
        replay = {
            1: torch.tensor([[-2.0, 0.0], [-1.8, 0.1]]),
            3: torch.tensor([[2.0, 0.0], [1.8, -0.1]]),
        }
        audit = consolidate_classifier(
            model, self.prototypes, 0.01, 80, 0.03, 20, 99, 'cpu',
            replay_embeddings=replay,
            replay_source='balanced_current_encoder_raw_replay',
            persistent_raw_example_count=4,
        )
        self.assertLess(audit['cross_entropy_after'], audit['cross_entropy_before'])
        self.assertEqual(audit['source'], 'balanced_current_encoder_raw_replay')
        self.assertEqual(audit['fit_sample_count'], 4)
        self.assertEqual(audit['synthetic_sample_count'], 0)
        self.assertEqual(audit['persistent_embedding_count'], 0)
        self.assertEqual(audit['persistent_raw_example_count'], 4)

    def test_task_class_bias_uses_small_calibrator_and_preserves_classifier(self):
        model = TopModel(2, 4, cosine=False)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.tensor([
                [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
            ]))
            model.classifier.bias.copy_(torch.tensor([2.0, 2.0, -2.0, -2.0]))
        replay = {
            0: torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
            1: torch.tensor([[-1.0, 0.0], [-0.9, -0.1]]),
            2: torch.tensor([[0.0, 1.0], [0.1, 0.9]]),
            3: torch.tensor([[0.0, -1.0], [-0.1, -0.9]]),
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.classifier.named_parameters()
        }
        audit = consolidate_task_class_bias(
            model, replay, {0: [0, 1], 1: [2, 3]},
            class_regularization=0.01,
            task_regularization=0.01,
            task_weight=1.3,
            steps=200,
            lr=0.03,
            samples_per_class=20,
            device='cpu',
            persistent_raw_example_count=8,
        )
        self.assertLess(audit['cross_entropy_after'], audit['cross_entropy_before'])
        self.assertEqual(audit['learned_parameter_count'], 8)
        self.assertFalse(audit['classifier_parameters_changed'])
        self.assertEqual(audit['persistent_embedding_count'], 0)
        for name, parameter in model.classifier.named_parameters():
            torch.testing.assert_close(parameter, before[name])
        rows = torch.cat([replay[class_id] for class_id in range(4)])
        labels = torch.arange(4).repeat_interleave(2)
        self.assertEqual(int((model(rows).argmax(1) == labels).sum()), 8)

    def test_task_class_bias_requires_exact_task_partition(self):
        model = TopModel(2, 4, cosine=False)
        replay = {
            0: torch.tensor([[1.0, 0.0]]),
            1: torch.tensor([[-1.0, 0.0]]),
        }
        with self.assertRaises(ValueError):
            consolidate_task_class_bias(
                model, replay, {0: [0]}, 0.01, 0.01, 1.3,
                10, 0.03, 20, 'cpu', persistent_raw_example_count=2,
            )

    def test_old_top_model_state_loads_without_calibration_buffers(self):
        source = TopModel(2, 4, cosine=False)
        old_state = {
            key: value for key, value in source.state_dict().items()
            if '_logit_calibration_' not in key
        }
        restored = TopModel(2, 4, cosine=False)
        restored.load_state_dict(old_state)
        self.assertFalse(bool(restored._logit_calibration_enabled))

    def test_cosine_scale_and_parameter_flags_are_preserved(self):
        torch.manual_seed(7)
        model = TopModel(2, 5, cosine=True)
        model.train()
        scale = model.scale.detach().clone()
        flags = {name: value.requires_grad for name, value in model.named_parameters()}
        audit = consolidate_classifier(
            model,
            self.prototypes,
            regularization=0.01,
            steps=30,
            lr=0.01,
            samples_per_class=3,
            seed=42,
            device='cpu',
        )
        self.assertTrue(model.training)
        torch.testing.assert_close(model.scale, scale)
        self.assertEqual(
            {name: value.requires_grad for name, value in model.named_parameters()},
            flags,
        )
        self.assertEqual(audit['class_count'], 2)

    def test_invalid_prototype_is_rejected(self):
        with self.assertRaises(ValueError):
            balanced_prototype_batch({}, 2, 1, 'cpu')
        with self.assertRaises(ValueError):
            balanced_prototype_batch(self.prototypes, 0, 1, 'cpu')


if __name__ == '__main__':
    unittest.main()
