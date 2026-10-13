import unittest
from pathlib import Path
import json
import tempfile

import replay11_matrix as matrix


class Replay11MatrixTest(unittest.TestCase):
    def test_completed_run_rejects_wrong_party_count(self):
        history = [
            {'step': 'event_0_CIL', 'per_task_accs': {'task_0': 0.8},
             'per_task_accs_taskil': {'task_0': 0.8}},
            {'step': 'event_1_CIL',
             'per_task_accs': {'task_0': 0.6, 'task_1': 0.7},
             'per_task_accs_taskil': {'task_0': 0.9, 'task_1': 0.8}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'checkpoints').mkdir()
            (run / 'checkpoints/event_1_CIL.pt').write_bytes(b'checkpoint')
            (run / 'results.json').write_text(json.dumps({
                'config': {'cl_method': 'ocar', 'seed': 42, 'num_tasks': 2},
                'task_acc_history': history,
            }))
            config = {'cl_method': 'ocar', 'seed': 42,
                      'num_tasks': 2, 'num_parties': 4,
                      'vector_npz': '/data/' + matrix.UPMC_FOUR_CACHE}
            (run / 'config.json').write_text(json.dumps(config))
            self.assertAlmostEqual(matrix.validate_completed_run(
                ('upmc_food101', 'ocar', 42), run,
                expected_tasks=2)['aa_final_cil'], 0.65)
            config['num_parties'] = 2
            (run / 'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                matrix.validate_completed_run(
                    ('upmc_food101', 'ocar', 42), run, expected_tasks=2)

    def test_success_record_rejects_modified_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            run = job_dir / 'run_1'
            (run / 'checkpoints').mkdir(parents=True)
            checkpoint = run / 'checkpoints/event_1_CIL.pt'
            checkpoint.write_bytes(b'checkpoint')
            (run / 'config.json').write_text(json.dumps({
                'cl_method': 'ocar', 'seed': 42, 'num_tasks': 2,
                'num_parties': 4,
                'vector_npz': '/data/' + matrix.UPMC_FOUR_CACHE,
            }))
            results = run / 'results.json'
            results.write_text(json.dumps({
                'config': {'cl_method': 'ocar', 'seed': 42, 'num_tasks': 2},
                'task_acc_history': [
                    {'step': 'event_0_CIL',
                     'per_task_accs': {'task_0': 0.8},
                     'per_task_accs_taskil': {'task_0': 0.8}},
                    {'step': 'event_1_CIL',
                     'per_task_accs': {'task_0': 0.6, 'task_1': 0.7},
                     'per_task_accs_taskil': {'task_0': 0.9, 'task_1': 0.8}},
                ],
            }))
            record = {
                'dataset': 'upmc_food101', 'method': 'ocar',
                'seed': 42, 'smoke': True,
                'data_hashes': matrix.expected_data('upmc_food101'),
                'metrics': matrix.validate_completed_run(
                    ('upmc_food101', 'ocar', 42), run, 2),
                'results_sha256': matrix.sha256_file(results),
                'checkpoint_sha256': matrix.sha256_file(checkpoint),
            }
            matrix.verify_success_record(job_dir, record)
            results.write_text(results.read_text() + '\n')
            with self.assertRaises(ValueError):
                matrix.verify_success_record(job_dir, record)

    def test_upmc_authority_binds_new_cache_and_clean_source(self):
        expected = matrix.expected_data('upmc_food101')
        self.assertEqual(expected[matrix.UPMC_FOUR_CACHE],
                         'dbdd3a2979203ba0ee804de7d3b128426f869b0a127e4b87ebbe023e5750023b')
        self.assertIn('upmc_food101_image_clean_v1/SELECTION_MANIFEST.json',
                      expected)
        self.assertNotIn('upmc_food101_image_clean_v1/upmc_food101_vfl.npz',
                         expected)

    def test_metrics_require_complete_class_il_history(self):
        history = [
            {'step': 'event_0_CIL', 'per_task_accs': {'task_0': 0.8},
             'per_task_accs_taskil': {'task_0': 0.8}},
            {'step': 'event_1_CIL',
             'per_task_accs': {'task_0': 0.6, 'task_1': 0.7},
             'per_task_accs_taskil': {'task_0': 0.9, 'task_1': 0.8}},
        ]
        metrics = matrix.metrics_from_history(history, expected_tasks=2)
        self.assertAlmostEqual(metrics['aa_final_cil'], 0.65)
        self.assertAlmostEqual(metrics['bwt_cil'], -0.2)
        with self.assertRaises(ValueError):
            matrix.metrics_from_history(history[:1], expected_tasks=2)

    def test_deferred_metrics_use_diagonal_and_final_row(self):
        history = [
            {'step': 'event_0_CIL', 'per_task_accs': {'task_0': 0.8}},
            {'step': 'event_1_CIL',
             'per_task_accs': {'task_0': 0.6, 'task_1': 0.7},
             'per_task_accs_taskil': {'task_0': 0.9, 'task_1': 0.8}},
        ]
        metrics = matrix.metrics_from_history(history, 2, deferred=True)
        self.assertEqual(set(metrics), {'aa_final_cil', 'bwt_cil',
                                        'task_il_final'})
        self.assertAlmostEqual(metrics['aa_final_cil'], 0.65)
        self.assertAlmostEqual(metrics['bwt_cil'], -0.2)
        with self.assertRaises(ValueError):
            matrix.metrics_from_history(history[:1], 2, deferred=True)

    def test_deferred_run_has_fixed_directory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run').mkdir()
            self.assertEqual(matrix.run_directories(root, smoke=False),
                             [root / 'run'])
            self.assertEqual(matrix.run_directories(root, smoke=True), [])

    def test_exact_99_cells_and_no_two_party_upmc(self):
        jobs = matrix.jobs()
        self.assertEqual(len(jobs), 99)
        self.assertEqual(len(set(jobs)), 99)
        self.assertEqual(set(matrix.METHODS), {
            'balanced_contrast', 'lpr', 'hpcr', 'ocar', 'alsar',
            'mir', 'icarl', 'er', 'der_pp', 'er_ace', 'adagauss',
        })
        command = matrix.command_for(
            ('upmc_food101', 'ocar', 42), Path('/tmp/replay11-test'),
            python=Path('/opt/python'), worktree=Path('/opt/worktree'),
            device='cuda:0')
        self.assertEqual(command[command.index('--num_parties') + 1], '4')
        self.assertIn('upmc_food101_image_clean_4party_v1',
                      command[command.index('--vector_npz') + 1])
        self.assertEqual(command[command.index('--cl_method') + 1], 'ocar')
        self.assertEqual(command[command.index('--formal_deferred_evaluation') + 1], '1')

    def test_summary_is_explicitly_labeled_pilot(self):
        self.assertEqual(matrix.PILOT_TABLE_NAME, 'REPLAY11_PILOT_TABLE.csv')

    def test_full_job_uses_deferred_test_access(self):
        command = matrix.command_for(
            ('upmc_food101', 'mir', 42), Path('/tmp/replay11-test'),
            python=Path('/opt/python'), worktree=Path('/opt/worktree'),
            device='cuda:0')
        self.assertEqual(command[command.index('--formal_deferred_evaluation') + 1], '1')
        self.assertEqual(command[command.index('--exp_name') + 1], 'run')

    def test_mir_uses_paper_sgd_and_full_head_with_dataset_budget(self):
        budgets = {'cifar100': '50', 'isolet': '50',
                   'upmc_food101': '20'}
        for dataset, epochs in budgets.items():
            with self.subTest(dataset=dataset):
                command = matrix.command_for(
                    (dataset, 'mir', 42), Path('/tmp/replay11-test'),
                    python=Path('/opt/python'), worktree=Path('/opt/worktree'),
                    device='cuda:0')
                options = dict(zip(command[2::2], command[3::2]))
                self.assertEqual(options['--task_ce_mode'], 'full')
                self.assertEqual(options['--optimizer'], 'sgd')
                self.assertEqual(float(options['--momentum']), 0.0)
                self.assertEqual(float(options['--weight_decay']), 0.0)
                self.assertEqual(options['--epochs_per_task'], epochs)

                er = matrix.command_for(
                    (dataset, 'er', 42), Path('/tmp/replay11-test'),
                    python=Path('/opt/python'), worktree=Path('/opt/worktree'),
                    device='cuda:0')
                er_options = dict(zip(er[2::2], er[3::2]))
                self.assertEqual(er_options['--epochs_per_task'], epochs)
                if dataset != 'cifar100':
                    self.assertEqual(er_options['--optimizer'], 'adamw')
                    self.assertEqual(er_options['--task_ce_mode'], 'current')

    def test_isolet_uses_host_local_cache_path(self):
        command = matrix.command_for(
            ('isolet', 'ocar', 42), Path('/tmp/replay11-test'),
            python=Path('/opt/python'), worktree=Path('/opt/worktree'),
            device='cuda:1', data_root=Path('/other-host/data'))
        self.assertEqual(command[command.index('--vector_npz') + 1],
                         str(Path('/other-host/data') / 'isolet/isolet_vfl.npz'))

    def test_deferred_provenance_covers_new_algorithms(self):
        from adaptive_consolidation_audit import FORMAL_SOURCE_FILES
        for method in ('alsar', 'balanced_contrast', 'hpcr', 'icarl',
                       'lpr', 'mir', 'ocar'):
            self.assertIn('cl_methods/' + method + '.py', FORMAL_SOURCE_FILES)

    def test_cifar_smoke_uses_two_tasks_without_custom_task_list(self):
        command = matrix.command_for(
            ('cifar100', 'er', 42), Path('/tmp/replay11-test'),
            python=Path('/opt/python'), worktree=Path('/opt/worktree'),
            device='cuda:0', smoke=True)
        self.assertEqual(command[command.index('--num_tasks') + 1], '2')
        self.assertEqual(command[command.index('--epochs_per_task') + 1], '1')
        self.assertEqual(command[command.index('--formal_deferred_evaluation') + 1], '0')

    def test_shards_partition_jobs_without_overlap(self):
        first = set(matrix.jobs_for_shard(0, 3))
        second = set(matrix.jobs_for_shard(1, 3))
        third = set(matrix.jobs_for_shard(2, 3))
        self.assertFalse(first & second or first & third or second & third)
        self.assertEqual(first | second | third, set(matrix.jobs()))


if __name__ == '__main__':
    unittest.main()
