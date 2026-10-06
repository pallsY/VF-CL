import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fair_main_table_3datasets as fair
from dataset_protocol_smoke import validation_command


class FairMainTableProtocolTest(unittest.TestCase):
    def test_matrix_has_two_new_datasets_eight_methods_three_seeds(self):
        self.assertEqual(48, len(fair.jobs()))
        self.assertEqual(set(fair.DATASETS), {'isolet', 'upmc_food101'})
        self.assertEqual(8, len(fair.METHODS))
        self.assertEqual((42, 43, 44), fair.SEEDS)

    def test_task_partitions_cover_every_class_once(self):
        for dataset, cfg in fair.DATASETS.items():
            flattened = [label for task in cfg['tasks'] for label in task]
            self.assertEqual(list(range(cfg['num_classes'])), flattened, dataset)
            self.assertEqual(cfg['num_classes'], len(set(flattened)), dataset)

    def test_equal_exemplar_budget_is_in_every_command(self):
        for dataset in fair.DATASETS:
            er = fair.base_command(f'{dataset}:er:42', 'cuda:0', Path('/tmp/x'))
            der = fair.base_command(f'{dataset}:der_pp:42', 'cuda:0', Path('/tmp/x'))
            budget = str(20 * fair.DATASETS[dataset]['num_classes'])
            self.assertEqual('20', er[er.index('--er_per_class') + 1])
            self.assertEqual(budget, der[der.index('--der_buffer_size') + 1])

    def test_no_test_fitted_calibration_or_selection(self):
        command = fair.base_command('upmc_food101:ours:44', 'cuda:1', Path('/tmp/x'))
        self.assertEqual('0', command[command.index('--bic_enabled') + 1])
        self.assertEqual('0', command[command.index('--lambda_validation_enabled') + 1])
        self.assertEqual('uniform', command[command.index('--expected_party_kd_variant') + 1])

    def test_ours_uses_one_frozen_head_consolidation_protocol(self):
        expected = {
            '--head_consolidation_enabled': '1',
            '--head_consolidation_mode': 'task_class_bias',
            '--head_consolidation_regularization': '0.01',
            '--head_consolidation_lr': '0.03',
            '--head_consolidation_steps': '600',
            '--head_consolidation_class_regularization': '0.01',
            '--head_consolidation_task_regularization': '0.01',
            '--head_consolidation_task_weight': '1.3',
            '--head_consolidation_samples_per_class': '20',
            '--head_consolidation_schedule': 'final',
        }
        for dataset in fair.DATASETS:
            command = fair.base_command(f'{dataset}:ours:42', 'cuda:0', Path('/tmp/x'))
            for flag, value in expected.items():
                self.assertEqual(value, command[command.index(flag) + 1])

    def test_dataset_training_parameters_are_shared_by_every_method(self):
        for dataset, cfg in fair.DATASETS.items():
            expected = {
                '--optimizer': cfg['optimizer'],
                '--lr': str(cfg['lr']),
                '--bottom_lr_scale': str(cfg['bottom_lr_scale']),
                '--weight_decay': str(cfg['weight_decay']),
                '--task_ce_mode': cfg['task_ce_mode'],
            }
            for method in fair.METHODS:
                command = fair.base_command(
                    f'{dataset}:{method}:42', 'cuda:0', Path('/tmp/x')
                )
                for flag, value in expected.items():
                    self.assertEqual(value, command[command.index(flag) + 1])

    def test_new_vector_protocols_use_current_task_ce(self):
        self.assertEqual('current', fair.DATASETS['isolet']['task_ce_mode'])
        self.assertEqual('current', fair.DATASETS['upmc_food101']['task_ce_mode'])

    def test_protocol_smoke_uses_training_validation(self):
        for dataset, cfg in fair.DATASETS.items():
            command = validation_command(dataset, 'cuda:0', Path('/tmp/protocol'))
            self.assertEqual(
                '1', command[command.index('--lambda_validation_enabled') + 1]
            )
            self.assertEqual(
                str(cfg['validation_per_class']),
                command[command.index('--lambda_validation_per_class') + 1],
            )
            self.assertEqual('0', command[command.index('--bic_enabled') + 1])
            self.assertEqual('3', command[command.index('--epochs_per_task') + 1])

    def test_selected_ce_is_bound_to_audited_report(self):
        report_path = fair.CE_PROTOCOL_SELECTION['report']
        self.assertTrue(report_path.is_file())
        self.assertEqual(
            fair.CE_PROTOCOL_SELECTION['sha256'], fair.file_sha256(report_path)
        )
        selection = __import__('json').loads(report_path.read_text(encoding='utf-8'))
        self.assertTrue(selection['passed'])
        self.assertFalse(selection['selection_test_used'])
        self.assertFalse(selection['ours_used_for_selection'])
        for dataset, cfg in fair.DATASETS.items():
            self.assertEqual(
                cfg['task_ce_mode'], selection['selected'][dataset]['ce_mode']
            )

    def test_worker_partition_is_complete_and_disjoint(self):
        left = set(fair.jobs(0, 2))
        right = set(fair.jobs(1, 2))
        self.assertFalse(left & right)
        self.assertEqual(set(fair.jobs()), left | right)

    def test_cifar_reuse_audit_hashes_every_frozen_artifact(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            artifacts = {}
            for name in fair.CIFAR100_REUSE:
                path = Path(source) / name
                path.write_text(name, encoding='utf-8')
                artifacts[name] = path
            with mock.patch.object(fair, 'CIFAR100_REUSE', artifacts):
                self.assertEqual(0, fair.audit_cifar_reuse(output))
            audit = Path(output) / 'cifar100_reuse' / 'AUDIT.json'
            self.assertTrue(audit.is_file())
            self.assertTrue((audit.parent / 'CIFAR100_REUSE_SUCCESS').is_file())


if __name__ == '__main__':
    unittest.main()
