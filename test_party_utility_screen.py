import copy
import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from data_utils import VFLDataset
from models import TopModel
from models import build_models
from vfl_trainer import VFLTrainer
from party_utility_screen import (
    replay_scores, validation_outcome, summarize_rows, screen_run,
)


class ReplayUtilityTests(unittest.TestCase):
    def test_helpful_party_gets_weight_and_identical_models_have_zero_drift(self):
        args = types.SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2)],
            num_parties=2, aggregation='concat', device='cpu',
        )
        old_top = TopModel(2, 2).eval()
        with torch.no_grad():
            old_top.classifier.weight.copy_(torch.tensor([[2., 0.], [0., 2.]]))
            old_top.classifier.bias.zero_()
        new_top = copy.deepcopy(old_top)
        with torch.no_grad():
            new_top.classifier.weight[0, 0] = 1.

        def trainer(top):
            return types.SimpleNamespace(
                bottoms=[torch.nn.Identity(), torch.nn.Identity()],
                top_model=top,
                _aggregate=lambda parts: torch.cat(parts, dim=1),
            )

        replay = {0: torch.ones(1, 2), 1: torch.ones(1, 2)}
        frozen = {0: [0.5, 0.5], 1: [0.5, 0.5]}
        changed = replay_scores(trainer(old_top), trainer(new_top),
                                replay, [0, 1], frozen, args)
        row = next(row for row in changed if row['class_id'] == 0)
        self.assertAlmostEqual(row['utility_weights'][0], 1.0)
        self.assertAlmostEqual(row['utility_weights'][1], 0.0)
        self.assertAlmostEqual(row['utility'], 1.0)
        self.assertAlmostEqual(row['uniform'], 0.5)

        identical = replay_scores(trainer(old_top), trainer(old_top),
                                  replay, [0, 1], frozen, args)
        self.assertTrue(all(row[key] == 0.0 for row in identical
                            for key in ('utility', 'uniform', 'frozen', 'shuffled')))
        with self.assertRaises(ValueError):
            replay_scores(trainer(old_top), trainer(new_top),
                          {0: replay[0]}, [0, 1], frozen, args)

    def test_validation_ce_uses_only_pretransition_classes(self):
        args = types.SimpleNamespace(
            data='tabvfl', party_col_ranges=[(0, 1), (1, 2)],
            num_parties=2, aggregation='concat', device='cpu',
        )
        old_top = TopModel(2, 3).eval()
        with torch.no_grad():
            old_top.classifier.weight.copy_(torch.tensor(
                [[2., 0.], [0., 2.], [100., 100.]],
            ))
            old_top.classifier.bias.zero_()
        new_top = copy.deepcopy(old_top)
        with torch.no_grad():
            new_top.classifier.weight[0, 0] = 1.

        def trainer(top):
            return types.SimpleNamespace(
                bottoms=[torch.nn.Identity(), torch.nn.Identity()],
                top_model=top,
                _aggregate=lambda parts: torch.cat(parts, dim=1),
            )

        batches = [(torch.ones(1, 2), torch.zeros(1, dtype=torch.long))]
        outcome = validation_outcome(trainer(old_top), trainer(new_top),
                                     batches, [0, 1], [0, 1, 2], args)
        self.assertAlmostEqual(outcome['old_ce'], 0.693147, places=5)
        self.assertAlmostEqual(outcome['new_ce'], 1.313262, places=5)
        self.assertAlmostEqual(outcome['delta_ce'], 0.620115, places=5)
        self.assertEqual(outcome['delta_cil_error'], 1.0)

    def test_summary_applies_prespecified_control_margin_by_boundary(self):
        rows = [
            {
                'boundary': boundary, 'class_id': class_id,
                'delta_ce': float(class_id),
                'utility': float(class_id),
                'uniform': float(2 - class_id),
                'frozen': float(2 - class_id),
                'shuffled': float(2 - class_id),
                'utility_fallback': False,
            }
            for boundary in (0, 1) for class_id in (0, 1, 2)
        ]
        summary = summarize_rows(rows)
        self.assertTrue(summary['passed'])
        self.assertAlmostEqual(summary['centered_spearman']['utility'], 1.0)
        self.assertAlmostEqual(summary['centered_spearman']['uniform'], -1.0)
        self.assertEqual(summary['positive_boundaries'], 2)
        self.assertEqual(summary['eligible_boundaries'], 2)
        self.assertAlmostEqual(summary['mean_abs_utility_frozen_score_gap'], 4 / 3)
        self.assertAlmostEqual(summary['mean_abs_utility_shuffled_score_gap'], 4 / 3)

    @unittest.skipIf(os.name == 'nt', 'existing manifest fsync requires POSIX')
    def test_screen_reads_adjacent_checkpoints_without_final_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'source'
            output = root / 'screen'
            (run / 'checkpoints').mkdir(parents=True)
            data = root / 'small.npz'
            np.savez(data, X=np.tile(np.eye(4, 2, dtype=np.float32), (5, 1)),
                     y=np.repeat(np.arange(4), 5),
                     train_idx=np.arange(20), test_idx=np.arange(4),
                     view_names=np.asarray(['a', 'b']),
                     range_lo=np.asarray([0, 1]), range_hi=np.asarray([1, 2]))
            config = dict(data='tabvfl', data_path=str(root), vector_npz=str(data),
                          num_parties=2, num_classes=4, model_type='mlp',
                          embed_dim=2, aggregation='concat', device='cpu',
                          deterministic=0, batch_size=4, num_workers=0,
                          bic_enabled=0, lambda_validation_enabled=1,
                          lambda_validation_per_class=1,
                          lambda_validation_split_seed=9, data_flow_audit=0,
                          custom_tasks='0,1|2|3', num_tasks=3, seed=42,
                          output_dir=str(run), cosine_head=False)
            (run / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            args = types.SimpleNamespace(**config)
            VFLDataset(args)
            bottoms, top = build_models(args)
            trainer = VFLTrainer(bottoms, top, args)
            for event, seen in ((0, [0, 1]), (1, [0, 1, 2])):
                torch.save({
                    'task_id': event,
                    'seen_task_classes': ({0: [0, 1]} if event == 0
                                          else {0: [0, 1], 1: [2]}),
                    'trainer_state': trainer.get_state(),
                    'cl_state': {
                        'head_raw_replay': {
                            c: torch.tensor([[1., 0.]]) for c in seen
                        },
                        'class_party_weights': {
                            c: [0.5, 0.5] for c in seen
                        },
                    },
                }, run / 'checkpoints' / f'event_{event}_CIL.pt')
            before = sorted(str(path.relative_to(run)) for path in run.rglob('*'))
            result = screen_run(run, output)
            after = sorted(str(path.relative_to(run)) for path in run.rglob('*'))
            self.assertEqual(before, after)
            self.assertEqual(len(result['rows']), 2)
            self.assertEqual(result['summary']['boundaries'], 1)
            self.assertTrue((output / 'screen.json').is_file())
            manifest_path = run / 'validation' / 'validation_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['ordered_indices'] = list(reversed(manifest['ordered_indices']))
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'source validation manifest'):
                screen_run(run, root / 'tampered_screen')


if __name__ == '__main__':
    unittest.main()
