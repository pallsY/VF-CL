import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from run_sampleonly_nme_readonly_check import nme_readout


class NmeReadonlyCheckTests(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            device='cpu', data='synthvfl', party_col_ranges=[(0, 1), (1, 2)],
            num_parties=2, num_classes=4, model_type='mlp', embed_dim=2,
            batch_size=8, aggregation='concat', pilot_holdout_per_class=2,
        )

    def _checkpoint(self, tmp_path, per_class=3, raw=None):
        torch.manual_seed(0)
        from models import build_models
        from vfl_trainer import VFLTrainer
        bottoms, top = build_models(self._args())
        trainer = VFLTrainer(bottoms, top, self._args())
        if raw is None:
            raw = {
                class_id: torch.rand(per_class, 2) + class_id * 2.0
                for class_id in range(4)
            }
        payload = {
            'trainer_state': trainer.get_state(),
            'cl_state': {'head_raw_replay': raw},
            'seen_task_classes': {0: [0, 1], 1: [2, 3]},
        }
        path = tmp_path / 'formal_final.pt'
        torch.save(payload, path)
        return path

    def test_nme_separates_well_clustered_classes_perfectly(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._checkpoint(Path(tmp))
            args = self._args()

            class FakeDataset:
                def holdout_loader(self, classes):
                    torch.manual_seed(1)
                    batches = []
                    for class_id in classes:
                        x = torch.rand(2, 2) * 0.05 + class_id * 2.0
                        y = torch.full((2,), class_id, dtype=torch.long)
                        batches.append((x, y))
                    return batches

            with patch('cl_methods.icarl.split_features',
                       side_effect=lambda x, _: [x[:, :1], x[:, 1:]]):
                result = nme_readout(args, FakeDataset(), checkpoint)
            self.assertEqual(result['readout_rule'],
                             'icarl_nearest_exemplar_mean')
            self.assertEqual(result['raw_replay_classes'], 4)
            self.assertEqual(result['metrics']['count'], 8)
            self.assertGreaterEqual(result['metrics']['cil'], 0.75)
            self.assertEqual(result['raw_replay_per_class_counts'],
                             {0: 3, 1: 3, 2: 3, 3: 3})

    def test_missing_classes_in_raw_replay_are_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            raw = {class_id: torch.rand(3, 2) for class_id in range(3)}
            path = self._checkpoint(Path(tmp), raw=raw)
            args = self._args()

            class FakeDataset:
                def holdout_loader(self, classes):
                    return [(torch.rand(2, 2), torch.zeros(2, dtype=torch.long))]

            with patch('cl_methods.icarl.split_features',
                       side_effect=lambda x, _: [x[:, :1], x[:, 1:]]):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    nme_readout(args, FakeDataset(), path)

    def test_mean_uses_icarl_class_means_not_mean_first(self):
        """nme_readout's class means must match ICaRLCL._class_means
        (normalize each sample's embedding, then average, then renormalize),
        not a mean-first rule. Class 0 gets two examples that embed to very
        different norms, so mean-then-normalize (pulled toward the
        large-norm embedding) and normalize-then-mean (equal weight on
        both) diverge; the expected value is computed by applying the
        checkpoint's own bottoms/aggregate to the raw samples, not derived
        by hand."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args()
            torch.manual_seed(0)
            from models import build_models
            from vfl_trainer import VFLTrainer
            bottoms, top = build_models(args)
            trainer = VFLTrainer(bottoms, top, args)

            raw = {
                0: torch.stack([torch.tensor([3.0, 4.0]),
                                torch.tensor([30.0, 40.0])]),
                1: torch.stack([torch.tensor([0.0, -1.0]),
                                torch.tensor([0.0, -1.0])]),
                2: torch.stack([torch.tensor([-1.0, 0.0]),
                                torch.tensor([-1.0, 0.0])]),
                3: torch.stack([torch.tensor([-1.0, -1.0]),
                                torch.tensor([-1.0, -1.0])]),
            }
            payload = {
                'trainer_state': trainer.get_state(),
                'cl_state': {'head_raw_replay': raw},
                'seen_task_classes': {0: [0, 1], 1: [2, 3]},
            }
            checkpoint = Path(tmp) / 'formal_final.pt'
            torch.save(payload, checkpoint)

            def split(x, _args):
                return [x[:, :1], x[:, 1:]]

            # Reference embedding of class 0's two raw samples, using the
            # checkpoint's actual (seeded, reproducible) bottoms/aggregate.
            with torch.no_grad():
                parts = split(raw[0], args)
                class0_embeddings = trainer._aggregate(
                    [trainer.bottoms[i](parts[i]) for i in range(2)])
            mean_then_normalize = F.normalize(class0_embeddings.mean(0), dim=0)
            normalize_then_mean = F.normalize(
                F.normalize(class0_embeddings, dim=1).mean(0), dim=0)
            self.assertFalse(torch.allclose(mean_then_normalize,
                                            normalize_then_mean, atol=1e-3))

            captured = {}
            from cl_methods.icarl import ICaRLCL
            original = ICaRLCL._class_means

            def spy(self):
                class_ids, means = original(self)
                captured['means'] = means.clone()
                captured['class_ids'] = class_ids.clone()
                return class_ids, means

            class FakeDataset:
                def holdout_loader(self, classes):
                    # Two (args.pilot_holdout_per_class) probes per class,
                    # reusing each class's own raw samples, so every task
                    # has a nonempty holdout cohort of the expected size.
                    return [(raw[c].clone(),
                            torch.full((2,), c, dtype=torch.long))
                           for c in classes]

            with patch('cl_methods.icarl.split_features', side_effect=split), \
                 patch.object(ICaRLCL, '_class_means', spy):
                nme_readout(args, FakeDataset(), checkpoint)

        class0_index = captured['class_ids'].tolist().index(0)
        actual = captured['means'][class0_index]
        self.assertTrue(torch.allclose(actual, normalize_then_mean, atol=1e-5))
        self.assertFalse(torch.allclose(actual, mean_then_normalize, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
