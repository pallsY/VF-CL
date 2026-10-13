import copy
import unittest

import torch
import torch.nn.functional as F
from types import SimpleNamespace

import cl_methods.hpcr as hpcr
from cl_methods.hpcr import HPCRCL, proxy_supcon_loss, similarity_distillation_loss
from models import TopModel


class HPCRLossTest(unittest.TestCase):
    def test_formal_readout_masks_never_seen_proxy_columns(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 3, cosine=True)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=3, er_per_class=2, er_batch=2,
            hpcr_temperature=1.0,
        )
        method = HPCRCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        with torch.no_grad():
            top.classifier.weight.copy_(torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        cached = ((torch.tensor([[1.0, 0.0], [0.6, 0.8]]),
                   torch.tensor([0, 1])),)
        self.assertTrue(hasattr(method, 'evaluate_class_il_readouts_cached'))
        readouts = method.evaluate_class_il_readouts_cached(
            cached, {0: [0, 1]})
        self.assertEqual(readouts['class_il_pred_task']['task_0'], 1.0)
        self.assertEqual(readouts['class_il_global']['task_0'], 0.5)
        self.assertEqual(readouts['class_il_pred_task_counts']['task_0'],
                         {'correct': 2, 'total': 2})
        from runner import _evaluate_cil_readouts_cached
        formal = _evaluate_cil_readouts_cached(
            method, trainer, cached, {0: [0, 1]}, 'cpu')
        self.assertEqual(formal['primary']['task_0'], 1.0)
        self.assertEqual(formal['companions']['class_il_global']['task_0'], 0.5)

    def test_stored_logits_exclude_future_class_targets(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 4, cosine=True)
        trainer = SimpleNamespace(bottoms=[bottom], top_model=top)
        args = SimpleNamespace(device='cpu', num_classes=4,
                               batch_size=1, er_per_class=2, er_batch=1)
        method = HPCRCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        method._insert(
            torch.tensor([[1.0, 0.0]]), torch.tensor([0]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[1.0, 0.0]]),
        )
        stored = method.memory[0][0][2]
        self.assertTrue(torch.isfinite(stored[:2]).all())
        self.assertTrue(torch.isneginf(stored[2:]).all())
        from runner import _checkpoint_values_equal
        self.assertTrue(_checkpoint_values_equal(
            method.get_state(), copy.deepcopy(method.get_state())))
        current = torch.tensor([[1.0, 2.0, 3.0, 4.0]],
                               requires_grad=True)
        loss = hpcr.proxy_logit_distillation_loss(
            current, stored[None, :], torch.tensor([0, 2]))
        loss.backward()
        self.assertEqual(float(current.grad[0, 2]), 0.0)

    def test_first_unreplayed_batch_learns_from_paper_contrast(self):
        bottom = torch.nn.Linear(1, 2)
        top = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0,
        )
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)],
            device='cpu', num_classes=2, er_per_class=2, er_batch=2,
        )
        method = HPCRCL(trainer, args)
        loss = method._hpcr_loss(
            [bottom], top, torch.tensor([[1.0], [2.0]]),
            torch.tensor([0, 1]))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(bottom.weight.grad.abs().sum()), 0.0)
        self.assertEqual(sum(map(len, method.memory.values())), 2)

    def test_paper_contrast_adds_nonself_sample_pairs_when_batch_large(self):
        features = F.normalize(torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]] * 5), dim=1)
        proxies = F.normalize(torch.tensor([[1.0, 0.2], [0.2, 1.0]]), dim=1)
        labels = torch.tensor([0, 0, 1, 1] * 5)
        tau = 0.5
        proxy_scores = features @ proxies[labels].T / tau
        sample_scores = features @ features.T / tau
        positive = labels[:, None].eq(labels[None, :])
        nonself = ~torch.eye(len(labels), dtype=torch.bool)
        expected = []
        for anchor in range(len(labels)):
            denominator = torch.logsumexp(torch.cat((
                proxy_scores[anchor], sample_scores[anchor, nonself[anchor]])), 0)
            positive_scores = sample_scores[anchor, positive[anchor] & nonself[anchor]]
            numerator = torch.logaddexp(proxy_scores[anchor, anchor].expand_as(
                positive_scores), positive_scores)
            expected.append((denominator - numerator).mean())
        actual = proxy_supcon_loss(features, proxies, labels, temperature=tau)
        torch.testing.assert_close(actual, torch.stack(expected).mean())

    def test_temperature_schedule_is_reciprocal_paper_equation(self):
        method = HPCRCL(SimpleNamespace(), SimpleNamespace(
            er_per_class=2, er_batch=2, hpcr_temperature=0.09,
            hpcr_temp_max=0.16, hpcr_temp_min=0.05))
        for step, expected in ((0, 0.5625), (125, 0.09 / 0.105), (250, 1.8)):
            method.step = step
            self.assertAlmostEqual(method._contrastive_scale(), expected, places=6)

    def test_legacy_matrix_ratio_bounds_resolve_to_paper_temperatures(self):
        method = HPCRCL(SimpleNamespace(), SimpleNamespace(
            er_per_class=2, er_batch=2, hpcr_temperature=0.09,
            hpcr_temp_max=1.8, hpcr_temp_min=0.5625))
        self.assertAlmostEqual(method.max_temp, 0.16)
        self.assertAlmostEqual(method.min_temp, 0.05)
        self.assertAlmostEqual(method._contrastive_scale(), 0.5625)

    def test_proxy_contrastive_gradient_reaches_features_and_proxies(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                                requires_grad=True)
        proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]],
                               requires_grad=True)
        loss = proxy_supcon_loss(features, proxies, torch.tensor([0, 0, 1]),
                                 temperature=0.09)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertGreater(float(proxies.grad.abs().sum()), 0.0)

    def test_pcd_weights_only_batch_selected_class_columns(self):
        self.assertTrue(hasattr(hpcr, 'proxy_logit_distillation_loss'))
        current = torch.tensor([[1.0, 2.0, 100.0]], requires_grad=True)
        stored = torch.zeros_like(current)
        loss = hpcr.proxy_logit_distillation_loss(
            current, stored, torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(loss.detach()), 6.0)
        loss.backward()
        self.assertEqual(float(current.grad[0, 2]), 0.0)

    def test_pcd_ignores_unseen_stored_logits_without_nan_gradient(self):
        current = torch.tensor([[1.0, 2.0]], requires_grad=True)
        stored = torch.tensor([[0.0, float('nan')]])
        loss = hpcr.proxy_logit_distillation_loss(
            current, stored, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertEqual(float(current.grad[0, 1]), 0.0)

    def test_scd_matches_paper_printed_probability_direction(self):
        current = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-0.3, 0.95]],
                               requires_grad=True)
        stored = torch.tensor([[0.8, 0.6], [-0.5, 0.866], [0.2, 0.98]])
        tau = 0.5
        p = F.normalize(current, dim=1) @ F.normalize(current, dim=1).T / tau
        q = F.normalize(stored, dim=1) @ F.normalize(stored, dim=1).T / tau
        diagonal = torch.eye(3, dtype=torch.bool)
        p = p.masked_fill(diagonal, -torch.inf)
        q = q.masked_fill(diagonal, -torch.inf)
        expected = -(F.softmax(p, dim=1) * F.log_softmax(q, dim=1)
                     ).masked_fill(diagonal, 0).sum(1).mean()
        actual = similarity_distillation_loss(current, stored, tau)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.0)

    def test_training_and_inference_use_bias_free_cosine_logits(self):
        bottom = torch.nn.Identity()
        top = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0)
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 2)], device='cpu',
            num_classes=2, er_per_class=2, er_batch=2,
            hpcr_temperature=0.5)
        method = HPCRCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        with torch.no_grad():
            top.classifier.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
            top.classifier.bias.fill_(100.0)
        inputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        expected = F.normalize(inputs, dim=1) @ F.normalize(
            top.classifier.weight, dim=1).T / 0.5
        torch.testing.assert_close(top(inputs), expected)
        logits, _ = method._forward([bottom], top, inputs)
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(top(inputs * 7), expected)
        reconstructed = TopModel(2, 2, cosine=True)
        reconstructed.load_state_dict(top.state_dict(), strict=True)
        torch.testing.assert_close(reconstructed(inputs), expected)

    def test_four_party_two_task_replay_and_restored_inference(self):
        torch.manual_seed(3)
        bottoms = [torch.nn.Linear(1, 2) for _ in range(4)]
        top = TopModel(8, 1)
        trainer = SimpleNamespace(
            bottoms=bottoms, top_model=top,
            _aggregate=lambda values: torch.cat(values, 1),
            bytes_transmitted=0, comm_rounds=0)
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2), (2, 3), (3, 4)],
            device='cpu', num_classes=4, er_per_class=1, er_batch=2,
            hpcr_min_samples=4)
        method = HPCRCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        task0 = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                              [0.0, 1.0, 0.0, 1.0]])
        method._hpcr_loss(bottoms, top, task0, torch.tensor([0, 1])).backward()
        method.before_task(1, [2, 3], [0, 1, 2, 3])
        task1 = torch.tensor([[2.0, 1.0, 2.0, 1.0],
                              [1.0, 2.0, 1.0, 2.0]])
        loss = method._hpcr_loss(bottoms, top, task1, torch.tensor([2, 3]))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(method.memory), 4)
        self.assertTrue(all(len(rows) <= 1 for rows in method.memory.values()))
        self.assertGreater(trainer.comm_rounds, 0)
        restored = HPCRCL(trainer, args)
        restored.load_state(method.get_state())
        self.assertEqual(restored.step, method.step)
        self.assertTrue(top.cosine)
        self.assertEqual(top(task1.new_ones((1, 8))).size(1), 4)

    def test_task_epochs_do_not_reinsert_the_same_stream(self):
        bottom = torch.nn.Linear(1, 2)
        top = TopModel(2, 2)
        trainer = SimpleNamespace(
            bottoms=[bottom], top_model=top,
            _aggregate=lambda values: values[0],
            bytes_transmitted=0, comm_rounds=0)
        args = SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1)], device='cpu',
            num_classes=2, er_per_class=2, er_batch=2)
        method = HPCRCL(trainer, args)
        method.before_task(0, [0, 1], [0, 1])
        method._steps_per_epoch = 1
        inputs = torch.tensor([[1.0], [2.0]])
        labels = torch.tensor([0, 1])
        method._hpcr_loss([bottom], top, inputs, labels)
        method._hpcr_loss([bottom], top, inputs, labels)
        self.assertEqual(method.seen_count, {0: 1, 1: 1})

    def test_buffer_caps_examples_and_restores_stored_targets(self):
        args = SimpleNamespace(er_per_class=2, er_batch=2, num_classes=3)
        first = HPCRCL(SimpleNamespace(), args)
        first._insert(torch.arange(3.0)[:, None], torch.tensor([0, 0, 0]),
                      torch.ones(3, 2), torch.arange(6.0).reshape(3, 2))
        self.assertEqual(len(first.memory[0]), 2)
        self.assertEqual(first.seen_count[0], 3)
        self.assertTrue(torch.isneginf(first.memory[0][0][2][2]))
        second = HPCRCL(SimpleNamespace(), args)
        second.load_state(first.get_state())
        self.assertEqual(len(second.memory[0]), 2)
        torch.testing.assert_close(second.memory[0][0][3],
                                   first.memory[0][0][3])


if __name__ == '__main__':
    unittest.main()
