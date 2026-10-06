import unittest
from types import SimpleNamespace

import torch

from data_utils import VFLDataset
from dataset_protocol_validation import validation_command
from ce_protocol_validation import jobs as ce_jobs
from ce_protocol_validation import validation_command as ce_validation_command
from runner import configure_task_ce
from selected_protocol_seed42 import discover_runs
from selected_protocol_seed42 import jobs as selected_jobs
from selected_protocol_seed42 import protocol_evidence
from selected_protocol_seed42 import validation_command as selected_validation_command
from vfl_trainer import VFLTrainer


class OptimizerProtocolTest(unittest.TestCase):
    def test_adamw_uses_scaled_bottom_lr_and_full_top_lr(self):
        bottom = torch.nn.Linear(3, 2)
        top = torch.nn.Linear(2, 4)
        args = SimpleNamespace(
            device='cpu', aggregation='concat', lr=0.003,
            bottom_lr_scale=0.25, optimizer='adamw', momentum=0.9,
            weight_decay=0.0001,
        )
        trainer = VFLTrainer([bottom], top, args)
        bottom_opts, top_opt = trainer._create_optimizers()
        self.assertIsInstance(bottom_opts[0], torch.optim.AdamW)
        self.assertIsInstance(top_opt, torch.optim.AdamW)
        self.assertAlmostEqual(0.00075, bottom_opts[0].param_groups[0]['lr'])
        self.assertAlmostEqual(0.003, top_opt.param_groups[0]['lr'])

    def test_historical_default_remains_sgd(self):
        args = SimpleNamespace(
            device='cpu', aggregation='concat', lr=0.001, momentum=0.9,
            weight_decay=0.0005,
        )
        trainer = VFLTrainer(
            [torch.nn.Linear(2, 2)], torch.nn.Linear(2, 2), args
        )
        bottom_opts, top_opt = trainer._create_optimizers()
        self.assertIsInstance(bottom_opts[0], torch.optim.SGD)
        self.assertIsInstance(top_opt, torch.optim.SGD)


class VectorValidationAuditTest(unittest.TestCase):
    def test_vector_selection_does_not_require_bic_manifest(self):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.args = SimpleNamespace(data='tabvfl')
        dataset.trainset = list(range(10))
        dataset.calibration_indices = set()
        dataset.calibration_manifest = None
        dataset.validation_indices = {1, 3}
        dataset.validation_manifest = {
            'sha256': 'a' * 64,
            'per_class': {'0': [1], '1': [3]},
        }
        audit = dataset.selection_audit()
        self.assertTrue(audit['passed'])
        self.assertIsNone(audit['calibration_manifest_sha256'])
        self.assertEqual('vector-train-validation', audit['evaluation_source'])
        self.assertFalse(audit['test_used_for_selection'])


class FullProtocolValidationTest(unittest.TestCase):
    def test_full_validation_uses_every_task_and_training_holdout(self):
        from fair_main_table_3datasets import DATASETS

        for dataset, cfg in DATASETS.items():
            command = validation_command(dataset, 'cuda:0', '/tmp/protocol')
            self.assertEqual(
                str(len(cfg['tasks'])),
                command[command.index('--num_tasks') + 1],
            )
            self.assertEqual(
                str(cfg['epochs_per_task']),
                command[command.index('--epochs_per_task') + 1],
            )
            self.assertEqual(
                '1', command[command.index('--lambda_validation_enabled') + 1]
            )


class TaskCEScopeTest(unittest.TestCase):
    def setUp(self):
        self.trainer = SimpleNamespace(ce_lo=7, ce_hi=9, ce_classes=[7, 8])

    def test_method_preserves_method_scope(self):
        configure_task_ce(self.trainer, 'method', [2, 3], [0, 1, 2, 3])
        self.assertEqual((7, 9), (self.trainer.ce_lo, self.trainer.ce_hi))
        self.assertEqual([7, 8], self.trainer.ce_classes)

    def test_current_scopes_to_new_classes(self):
        configure_task_ce(self.trainer, 'current', [2, 3], [0, 1, 2, 3])
        self.assertEqual((2, 4), (self.trainer.ce_lo, self.trainer.ce_hi))
        self.assertIsNone(self.trainer.ce_classes)

    def test_seen_scopes_to_all_seen_classes(self):
        configure_task_ce(self.trainer, 'seen', [2, 3], [0, 1, 2, 3])
        self.assertEqual((0, 4), (self.trainer.ce_lo, self.trainer.ce_hi))
        self.assertIsNone(self.trainer.ce_classes)

    def test_full_resets_method_scope(self):
        configure_task_ce(self.trainer, 'full', [2, 3], [0, 1, 2, 3])
        self.assertEqual((0, None), (self.trainer.ce_lo, self.trainer.ce_hi))
        self.assertIsNone(self.trainer.ce_classes)

    def test_noncontiguous_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            configure_task_ce(self.trainer, 'seen', [2], [0, 2])
        self.assertEqual((7, 9, [7, 8]),
                         (self.trainer.ce_lo, self.trainer.ce_hi, self.trainer.ce_classes))


class CEProtocolMatrixTest(unittest.TestCase):
    def test_matrix_has_two_datasets_three_modes_three_reference_methods(self):
        self.assertEqual(18, len(ce_jobs()))
        left, right = set(ce_jobs(0, 2)), set(ce_jobs(1, 2))
        self.assertFalse(left & right)
        self.assertEqual(set(ce_jobs()), left | right)

    def test_matrix_uses_fixed_training_validation_only(self):
        from fair_main_table_3datasets import DATASETS

        for spec in ce_jobs():
            dataset, mode, _, _ = spec.split(':')
            command = ce_validation_command(spec, 'cuda:0', '/tmp/ce')
            self.assertEqual(
                mode, command[command.index('--task_ce_mode') + 1]
            )
            self.assertEqual(
                '1', command[command.index('--lambda_validation_enabled') + 1]
            )
            self.assertEqual(
                str(DATASETS[dataset]['validation_split_seed']),
                command[command.index('--lambda_validation_split_seed') + 1],
            )


class SelectedProtocolSeed42Test(unittest.TestCase):
    def test_matrix_has_two_datasets_and_all_eight_methods(self):
        from fair_main_table_3datasets import METHODS

        self.assertEqual(16, len(selected_jobs()))
        self.assertEqual(8, len(METHODS))
        left, right = set(selected_jobs(0, 2)), set(selected_jobs(1, 2))
        self.assertFalse(left & right)
        self.assertEqual(set(selected_jobs()), left | right)

    def test_commands_use_frozen_ce_and_training_validation(self):
        from fair_main_table_3datasets import DATASETS

        for spec in selected_jobs():
            dataset, _, _ = spec.split(':')
            command = selected_validation_command(spec, 'cuda:0', '/tmp/selected')
            self.assertEqual(
                DATASETS[dataset]['task_ce_mode'],
                command[command.index('--task_ce_mode') + 1],
            )
            self.assertEqual(
                '1', command[command.index('--lambda_validation_enabled') + 1]
            )
            evidence = protocol_evidence(dataset)
            self.assertTrue(evidence['passed'])
            self.assertEqual(evidence['expected_sha256'], evidence['actual_sha256'])

    def test_run_discovery_uses_config_not_experiment_prefix(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'selected_protocol_seed42_timestamp'
            run.mkdir()
            (run / 'config.json').write_text('{}', encoding='utf-8')
            self.assertEqual([run], discover_runs(directory))


if __name__ == '__main__':
    unittest.main()
