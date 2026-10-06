import json
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from external_baseline_matrix import (
    audit_run,
    calibrated_trajectory,
    claim_next_job,
    command_for,
    expected_config,
    find_complete,
    find_incomplete,
    job_specs,
    recorded_code_commit,
    seed44_job_specs,
)


class ExternalBaselineMatrixTests(unittest.TestCase):
    def test_dynamic_claims_cover_each_job_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            claim_root = Path(directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                claimed = list(pool.map(
                    lambda _: claim_next_job(claim_root),
                    range(len(job_specs())),
                ))
            self.assertCountEqual(claimed, job_specs())
            self.assertIsNone(claim_next_job(claim_root))

    def test_launcher_uses_dynamic_claims_instead_of_static_partitioning(self):
        launcher = Path('run_external_baselines.sh').read_text()
        self.assertIn('external_baseline_matrix.py" claim', launcher)
        self.assertNotIn('index % workers', launcher)

    def test_launcher_prints_the_same_32_job_contract(self):
        output = subprocess.check_output(
            ['bash', 'run_external_baselines.sh', '--print-jobs'], text=True
        ).splitlines()
        self.assertEqual(output, job_specs())

    def test_matrix_has_32_unique_jobs_and_frozen_axes(self):
        jobs = job_specs()
        self.assertEqual(len(jobs), 32)
        self.assertEqual(len(set(jobs)), 32)
        self.assertEqual(
            {job.split(':')[0] for job in jobs},
            {'cifar100', 'tinyimagenet'},
        )
        self.assertEqual({int(job.split(':')[2]) for job in jobs}, {42, 43})

    def test_command_freezes_shared_protocol_and_uniform_variant(self):
        command = command_for(
            'tinyimagenet:proto_uniform:43', Path('/matrix'),
            Path('/repo'), Path('/python'),
        )
        joined = ' '.join(str(value) for value in command)
        for fragment in (
            '--num_classes 200', '--num_tasks 10', '--classes_per_task 20',
            '--num_parties 4', '--aggregation sum', '--epochs_per_task 50',
            '--batch_size 64', '--deterministic 1', '--num_workers 2',
            '--bic_enabled 1', '--bic_per_class 25',
            '--bic_split_seed 20260722', '--bic_steps 1000',
            '--save_task_checkpoints 2', '--seed 43',
            '--dep_tracking_enabled 1', '--party_kd_enabled 1',
            '--party_kd_mode uniform', '--expected_party_kd_variant uniform',
            '--bic_fit_mode joint_each_stage',
        ):
            self.assertIn(fragment, joined)

    def test_seed44_uniform_command_does_not_expand_frozen_matrix(self):
        command = command_for(
            'cifar100:proto_uniform:44', Path('/matrix'),
            Path('/repo'), Path('/python'),
        )
        joined = ' '.join(str(value) for value in command)
        self.assertIn('--seed 44', joined)
        self.assertIn('--bic_fit_mode joint_each_stage', joined)
        self.assertNotIn('cifar100:proto_uniform:44', job_specs())

    def test_seed44_external_contract_has_exactly_seven_missing_methods(self):
        jobs = seed44_job_specs()
        self.assertEqual(len(jobs), 7)
        self.assertEqual(len(set(jobs)), 7)
        self.assertEqual({job.split(':')[0] for job in jobs}, {'cifar100'})
        self.assertEqual({int(job.split(':')[2]) for job in jobs}, {44})
        self.assertEqual(
            {job.split(':')[1] for job in jobs},
            {'er', 'lwf', 'der_pp', 'ewc', 'gpm', 'fedprotip_vfl', 'finetune'},
        )

    def test_seed44_external_commands_use_rolling_checkpoint_mode(self):
        external = ' '.join(str(value) for value in command_for(
            'cifar100:er:44', Path('/matrix'), Path('/repo'), Path('/python')
        ))
        current_method = ' '.join(str(value) for value in command_for(
            'cifar100:proto_uniform:44', Path('/matrix'),
            Path('/repo'), Path('/python')
        ))
        self.assertIn('--save_task_checkpoints 3', external)
        self.assertIn('--save_task_checkpoints 2', current_method)

    def test_find_incomplete_requires_matching_config_and_resume_checkpoint(self):
        job = 'cifar100:er:44'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs' / 'external_cifar100_er_seed44_20260728_210000'
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps(expected_config(job)))
            (run / 'checkpoints' / 'resume_latest.pt').write_bytes(b'checkpoint')
            self.assertEqual(find_incomplete(job, root), run)
            (run / 'checkpoints' / 'event_9_CIL.pt').write_bytes(b'checkpoint')
            (run / 'results.json').write_text(json.dumps({
                'cl_metrics': {'AA_final': 0.1, 'AA_cil': 0.2, 'BWT': -0.1},
            }))
            with self.assertRaises(FileNotFoundError):
                find_incomplete(job, root)

    def test_corrupt_results_are_not_complete_and_can_resume_final_checkpoint(self):
        job = 'cifar100:er:44'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs' / 'external_cifar100_er_seed44_20260728_220000'
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'config.json').write_text(json.dumps(expected_config(job)))
            (run / 'checkpoints' / 'event_9_CIL.pt').write_bytes(b'checkpoint')
            (run / 'results.json').write_text('{"truncated":')

            with self.assertRaises(FileNotFoundError):
                find_complete(job, root)
            self.assertEqual(find_incomplete(job, root), run)

    def test_seed44_queue_contains_resume_and_dynamic_claim_contract(self):
        launcher = Path('run_cifar100_external_seed44.sh').read_text()
        self.assertIn('seed44-claim', launcher)
        self.assertIn('find-incomplete', launcher)
        self.assertIn('--resume_run_dir', launcher)
        self.assertIn('ulimit -Sn 65536', launcher)
        self.assertIn('formal_cifar100_metrics.py', launcher)
        self.assertIn('protocol_digest.json', launcher)

    def test_2080_launchers_do_not_reference_the_3080_server(self):
        contents = [
            Path(path).read_text()
            for path in (
                'external_baseline_matrix.py',
                'run_external_baselines.sh',
                'run_cifar100_external_seed44.sh',
                'run_cifar100_uniform_joint_seed44.sh',
            )
        ]
        self.assertTrue(all('/home/c3080/' not in text for text in contents))

    def test_seed44_queue_stops_globally_on_both_audit_paths(self):
        launcher = Path('run_cifar100_external_seed44.sh').read_text()
        audit_guard = 'if ! "$PY" "$ROOT/external_baseline_matrix.py" audit'
        self.assertEqual(launcher.count(audit_guard), 2)

    def test_gpu_memory_filter_uses_numeric_comparison(self):
        for path in (
            'run_external_baselines.sh',
            'run_cifar100_external_seed44.sh',
            'run_cifar100_uniform_joint_seed44.sh',
        ):
            launcher = Path(path).read_text()
            self.assertIn('($2 + 0) >= (minimum + 0)', launcher, path)

    def test_seed44_launcher_refreshes_the_formal_table(self):
        launcher = Path('run_cifar100_uniform_joint_seed44.sh').read_text()
        self.assertIn('JOB=cifar100:proto_uniform:44', launcher)
        self.assertIn('external_baseline_matrix.py" command', launcher)
        self.assertIn('formal_cifar100_metrics.py', launcher)
        self.assertIn('CIFAR100_PROTO_UNIFORM_SEED44_SUCCESS', launcher)

    def test_fedprotip_command_keeps_joint_final_and_predicted_task_primary(self):
        command = command_for(
            'cifar100:fedprotip_vfl:42', Path('/matrix'),
            Path('/repo'), Path('/python'),
        )
        joined = ' '.join(str(value) for value in command)
        self.assertIn('--cl_method fedprotip_vfl', joined)
        self.assertIn('--bic_fit_mode joint_final', joined)
        self.assertNotIn('--expected_party_kd_variant uniform', joined)

    def test_calibrated_trajectory_computes_final_aa_cil_and_bwt(self):
        history = [
            {'step': 'event_0_CIL', 'paired': {'calibrated': {
                'overall_accuracy': 0.6,
                'per_task_accuracy': {'task_0': 0.6},
            }}},
            {'step': 'event_1_CIL', 'paired': {'calibrated': {
                'overall_accuracy': 0.5,
                'per_task_accuracy': {'task_0': 0.4, 'task_1': 0.6},
            }}},
        ]
        result = calibrated_trajectory(history)
        self.assertEqual(result['AA_final'], 0.5)
        self.assertEqual(result['AA_cil'], 0.55)
        self.assertAlmostEqual(result['BWT'], -0.2)
        self.assertEqual(result['final_per_task'], {'task_0': 0.4, 'task_1': 0.6})

    def test_audit_writes_digest_and_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            job = 'cifar100:finetune:42'
            command = command_for(job, Path('/matrix'), Path('/repo'), Path('/python'))
            args = command[2:]
            config = {}
            for index in range(0, len(args), 2):
                config[args[index].removeprefix('--')] = args[index + 1]
            for key in ('num_classes', 'num_tasks', 'classes_per_task', 'num_parties',
                        'epochs_per_task', 'batch_size', 'num_workers', 'bic_enabled',
                        'bic_per_class', 'bic_split_seed', 'bic_steps',
                        'save_task_checkpoints', 'seed', 'deterministic',
                        'data_flow_audit'):
                config[key] = int(config[key])
            config['unlearn_after_tasks'] = [99]
            config['unlearn_classes'] = [[0]]
            (run_dir / 'config.json').write_text(json.dumps(config))
            (run_dir / 'results.json').write_text(json.dumps({
                'cl_metrics': {'AA_final': 0.1, 'AA_cil': 0.2, 'BWT': -0.1},
                'task_acc_history': [],
                'timing': [],
                'comm_stats': [],
                'calibration_audit': {'passed': True, 'test_used_for_fit': False},
            }))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                audit_run(job, run_dir, 'abc123')
            checkpoint = run_dir / 'checkpoints' / 'event_9_CIL.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')
            record = audit_run(job, run_dir, 'abc123')
            self.assertEqual(record['job'], job)
            self.assertTrue((run_dir / 'protocol_digest.json').is_file())

    def test_recorded_commit_preserves_existing_run_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / 'protocol_digest.json').write_text(json.dumps({
                'job': 'cifar100:finetune:42',
                'code_commit': 'old-commit',
            }))
            self.assertEqual(
                recorded_code_commit('cifar100:finetune:42', run_dir),
                'old-commit',
            )
            with self.assertRaisesRegex(ValueError, 'job'):
                recorded_code_commit('cifar100:finetune:43', run_dir)


if __name__ == '__main__':
    unittest.main()
