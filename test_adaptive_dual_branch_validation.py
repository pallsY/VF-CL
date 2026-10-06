import contextlib
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import adaptive_dual_branch_validation as driver
from metrics import MetricsTracker


DATASETS = ('cifar100', 'isolet', 'upmc_food101')
METRICS = ('AA_final', 'BWT', 'AA_final_taskil')


def canonical_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
    ).encode('ascii')).hexdigest()


def fake_incumbent(dataset, value=0.5):
    cell = {'cifar100': 'B', 'isolet': 'A', 'upmc_food101': 'A'}[dataset]
    payload = {
        'spec': f'{dataset}:{cell}:42',
        'dataset': dataset,
        'cell': cell,
        'seed': 42,
        'source_paths': {'run': f'/frozen/{dataset}/{cell}'},
        'source_sha256': {'config': 'a' * 64, 'results': 'b' * 64,
                          'validation_manifest': 'c' * 64,
                          'final_checkpoint': 'd' * 64,
                          'head_audit': 'e' * 64, 'provenance': 'f' * 64},
        'validation_hash': driver.EXPECTED_VALIDATION_HASHES[dataset],
        'metrics': {metric: value for metric in METRICS},
        'checks': {'immutable_factorial_audit': True},
        'passed': True,
    }
    return {**payload, 'record_sha256': canonical_sha256(payload)}


def command_flags(command):
    start = next(
        index for index, value in enumerate(command) if value.startswith('--')
    )
    return dict(zip(command[start::2], command[start + 1::2]))


class ContractTest(unittest.TestCase):
    def test_deployment_paths_follow_factorial_resolution(self):
        self.assertEqual(driver.REPO, driver.factorial.ROOT)
        self.assertEqual(driver.WORKTREE, Path(driver.__file__).resolve().parent)
        self.assertEqual(driver.PYTHON, Path(driver.sys.executable).resolve())

    def test_primary_matrix_is_exactly_three_seed42_jobs(self):
        self.assertEqual(driver.DATASETS, DATASETS)
        self.assertEqual(driver.SEED, 42)
        self.assertEqual(driver.primary_specs(), [
            'cifar100:adaptive:42',
            'isolet:adaptive:42',
            'upmc_food101:adaptive:42',
        ])
        for seed in (41, 43, 44):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                driver.validate_job('cifar100', seed)

    def test_incumbents_metrics_and_tolerance_are_frozen(self):
        self.assertEqual(driver.INCUMBENT_CELLS, {
            'cifar100': 'B', 'isolet': 'A', 'upmc_food101': 'A',
        })
        self.assertEqual(driver.METRICS, METRICS)
        self.assertEqual(driver.TOLERANCE, Decimal('0.01'))
        self.assertEqual(set(driver.EXPECTED_VALIDATION_HASHES), set(DATASETS))
        self.assertTrue(all(
            len(value) == 64 for value in driver.EXPECTED_VALIDATION_HASHES.values()
        ))

    def test_only_two_pre_registered_ablations_exist(self):
        self.assertEqual(driver.ABLATIONS, {
            'fixed_half_ablation': 'fixed_half_ablation',
            'sample_mean_nll': 'sample_mean_ablation',
        })
        self.assertEqual(driver.ablation_specs(), [
            f'{dataset}:{name}:42'
            for name in driver.ABLATIONS for dataset in DATASETS
        ])

    def test_primary_commands_freeze_adaptive_protocol(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                command = driver.build_command(
                    dataset, '__DEVICE__', Path('/matrix') / dataset,
                )
                flags = command_flags(command)
                self.assertEqual(Path(command[0]), driver.PYTHON)
                self.assertEqual(Path(command[1]), driver.WORKTREE / 'main.py')
                self.assertEqual(flags['--seed'], '42')
                self.assertEqual(flags['--head_consolidation_enabled'], '1')
                self.assertEqual(
                    flags['--head_consolidation_mode'], 'adaptive_dual_branch',
                )
                self.assertEqual(flags['--head_consolidation_schedule'], 'final')
                self.assertEqual(flags['--head_full_lr'], '0.01')
                self.assertEqual(flags['--head_full_steps'], '500')
                self.assertEqual(flags['--head_bias_lr'], '0.03')
                self.assertEqual(flags['--head_bias_steps'], '600')
                self.assertEqual(flags['--head_gate_rule'], 'class_balanced')
                self.assertEqual(flags['--head_gate_solver_tolerance'], '1e-12')
                self.assertEqual(flags['--head_gate_solver_max_iterations'], '80')
                self.assertEqual(flags['--lambda_validation_enabled'], '1')
                self.assertEqual(
                    flags['--lambda_validation_split_seed'],
                    {'cifar100': '20260729', 'isolet': '20260809',
                     'upmc_food101': '20260809'}[dataset],
                )
                self.assertEqual(flags['--data_flow_audit'], '1')
                self.assertEqual(flags['--save_task_checkpoints'], '3')
                self.assertNotIn('--resume_run_dir', flags)

    def test_ablation_commands_change_only_gate_rule_identity_and_output(self):
        primary = driver.build_command('isolet', '0', Path('/primary'))
        ignored = {'--head_gate_rule', '--results_dir', '--exp_name'}
        primary_flags = command_flags(primary)
        for name, rule in driver.ABLATIONS.items():
            with self.subTest(name=name):
                command = driver.build_command(
                    'isolet', '0', Path('/ablations') / name,
                    ablation=name,
                )
                flags = command_flags(command)
                self.assertEqual(command[1], '-c')
                self.assertIn('_execute_ablation_main', command[2])
                self.assertIn(repr(name), command[2])
                self.assertEqual(flags['--head_gate_rule'], rule)
                self.assertEqual(
                    {key: value for key, value in flags.items() if key not in ignored},
                    {key: value for key, value in primary_flags.items()
                     if key not in ignored},
                )


class LiveIncumbentPreflightTest(unittest.TestCase):
    def test_live_incumbents_accept_only_missing_post_source_defaults(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                evidence = driver.incumbent_evidence(dataset)
                self.assertTrue(evidence['passed'])
                self.assertTrue(evidence['checks']['effective_config'])
                self.assertEqual(len(evidence['checks']), 13)
                self.assertTrue(all(evidence['checks'].values()))

    def test_live_cifar_incumbent_is_bound_to_exact_factorial_rerun(self):
        run = driver.factorial.REFERENCE_RUNS[('cifar100', 'B')]
        evidence = driver.incumbent_evidence('cifar100')
        self.assertEqual(evidence['source_paths']['run'], str(run))
        self.assertEqual(
            driver.factorial.REFERENCE_DECLARED_RUNS[('cifar100', 'B')],
            Path('/home/chase/Yangxx/VF-CL') /
            driver.factorial.REFERENCE_RELATIVE_RUNS[('cifar100', 'B')],
        )
        self.assertEqual(evidence['source_sha256'], {
            'config': '8da359a7fa3188e16a64feaf71ecf63bd5234d5d56bbf6b0e3be000cab44279f',
            'results': '2349f80862b11cd48a0af76ba09d850be61390c670fbb20351f398c2d12b3019',
            'validation_manifest': '7934fcbc5385883cd4312d7416d70dbae8a73044802431f766ee6de2249e5418',
            'final_checkpoint': 'ab7b84e7ba96766ead1838aa0b52e5ea877d69d2a4476484afe09478b649bdd8',
            'head_audit': '71840fa517a9c5d614526656e349a3c14d0466e5c77a573f147c38ffc47dc9f0',
            'data_flow_audit': '65f47b841c1253623df684f70793dec42c8c207100ccb8013fc976c065f310f9',
            'provenance': '594c1842570a731dc1385c6dcbfe62eb2a865722edfaaa8b1d8c20364cda58cc',
            'launch_started': '8d8ee990c89e9ad5dbf129fdff128a96a57f0609d0ca41be230557823c6d3589',
            'record': '0303fc8c890bb7c828e87b72c13035b1ebd862c55d16e6dab820ebec598f07be',
            'code_commit': '078da3cfb3af22383003cdbe763dfd7c3389efdc688fc5a72070a906622caf87',
            'matrix_identity': '4d612317ecf0129396be1dc18e267123a3e9629dc542ade2a28099d2a6111500',
        })
        self.assertTrue(all(evidence['checks'][name] for name in (
            'reference_artifacts', 'source_commit', 'provenance_contract',
        )))

    def test_live_cifar_incumbent_rejects_changed_declared_suffix(self):
        declared = dict(driver.factorial.REFERENCE_DECLARED_RUNS)
        run = declared[('cifar100', 'B')]
        declared[('cifar100', 'B')] = run.with_name(f'{run.name}-changed')
        with mock.patch.object(
                driver.factorial, 'REFERENCE_DECLARED_RUNS', declared), \
                self.assertRaisesRegex(ValueError, 'incumbent audit failed'):
            driver.incumbent_evidence('cifar100')

    def test_live_cifar_incumbent_rejects_changed_dataset_cell_role(self):
        declared = dict(driver.factorial.REFERENCE_DECLARED_RUNS)
        declared[('cifar100', 'B')] = declared[('cifar100', 'A')]
        with mock.patch.object(
                driver.factorial, 'REFERENCE_DECLARED_RUNS', declared), \
                self.assertRaisesRegex(ValueError, 'incumbent audit failed'):
            driver.incumbent_evidence('cifar100')

    def test_live_cifar_incumbent_rejects_changed_implementation_commit(self):
        with mock.patch.object(
                driver.factorial, '_CIFAR_B_IMPLEMENTATION_COMMIT', '0' * 40), \
                self.assertRaisesRegex(ValueError, 'incumbent audit failed'):
            driver.incumbent_evidence('cifar100')

    def test_live_cifar_incumbent_rejects_changed_artifact_byte(self):
        source = driver.factorial.REFERENCE_RUNS[('cifar100', 'B')]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / source.name
            run.mkdir()
            (run / 'config.json').write_bytes(
                (source / 'config.json').read_bytes() + b' ',
            )
            for relative in ('results.json', 'validation/validation_manifest.json'):
                target = run / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source / relative)
            for name in ('checkpoints', 'head_consolidation'):
                (run / name).symlink_to(source / name, target_is_directory=True)
            references = dict(driver.factorial.REFERENCE_RUNS)
            references[('cifar100', 'B')] = run
            with mock.patch.object(
                    driver.factorial, 'REFERENCE_RUNS', references), \
                    self.assertRaisesRegex(ValueError, 'incumbent audit failed'):
                driver.incumbent_evidence('cifar100')

    def test_live_cifar_a_does_not_borrow_cifar_b_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            evidence = driver.factorial.audit_reference('cifar100', 'A', root)
        self.assertIsNone(evidence['source_paths']['provenance'])
        self.assertFalse(evidence['checks']['provenance_contract'])
        self.assertFalse(evidence['passed'])


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260815_120000'
        )
        self.incumbents = {
            dataset: fake_incumbent(dataset) for dataset in DATASETS
        }
        self.incumbent_patch = mock.patch.object(
            driver, 'incumbent_evidence',
            side_effect=lambda dataset: json.loads(json.dumps(
                self.incumbents[dataset]
            )),
        )
        self.source_patch = mock.patch.object(
            driver, 'adaptive_source_identity',
            return_value={
                'commit': driver.SOURCE_COMMIT,
                'sha256': {'adaptive_consolidation_audit.py': '1' * 64},
            },
        )
        self.incumbent_patch.start()
        self.source_patch.start()

    def tearDown(self):
        self.source_patch.stop()
        self.incumbent_patch.stop()
        self.temporary.cleanup()

    def test_plan_freezes_all_commands_hashes_and_incumbents_before_launch(self):
        payload = driver.plan(self.root)
        self.assertEqual(payload['kind'], 'adaptive_primary_plan')
        self.assertEqual(payload['source_commit'], driver.SOURCE_COMMIT)
        self.assertEqual(list(payload['jobs']), list(DATASETS))
        self.assertEqual(set(payload['incumbents']), set(DATASETS))
        self.assertEqual(
            payload['incumbents'], self.incumbents,
        )
        for dataset in DATASETS:
            job = payload['jobs'][dataset]
            self.assertEqual(job['spec'], f'{dataset}:adaptive:42')
            self.assertEqual(job['command'][job['command'].index('--device') + 1],
                             '__DEVICE__')
            self.assertEqual(job['command_sha256'], canonical_sha256(job['command']))
            self.assertEqual(job['validation_hash'],
                             driver.EXPECTED_VALIDATION_HASHES[dataset])
        stored = json.loads((self.root / 'PRIMARY_PLAN.json').read_text())
        self.assertEqual(stored, payload)
        self.assertFalse((self.root / 'primary').exists())

    def test_plan_is_exclusive_idempotent_and_rejects_retroactive_artifacts(self):
        first = driver.plan(self.root)
        before = (self.root / 'PRIMARY_PLAN.json').stat()
        self.assertEqual(driver.plan(self.root), first)
        after = (self.root / 'PRIMARY_PLAN.json').stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns),
                         (after.st_ino, after.st_mtime_ns))

        other = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260815_120001'
        )
        run = other / 'primary' / 'cifar100' / 'outputs' / 'already_complete'
        run.mkdir(parents=True)
        (run / 'results.json').write_text('{}')
        with self.assertRaises(ValueError):
            driver.plan(other)
        self.assertFalse((other / 'PRIMARY_PLAN.json').exists())

    def test_plan_rejects_wrong_basename_outside_and_symlink_ancestor(self):
        with self.assertRaises(ValueError):
            driver.plan(Path(self.temporary.name) / 'wrong')
        if not hasattr(os, 'symlink'):
            return
        real = Path(self.temporary.name) / 'real'
        real.mkdir()
        link = Path(self.temporary.name) / 'link'
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            self.skipTest('symlinks unavailable')
        with self.assertRaises(ValueError):
            driver.plan(link / 'adaptive_dual_branch_validation_seed42_20260815_120002')

    def test_stored_plan_is_bound_to_fresh_source_and_factorial_incumbents(self):
        payload = driver.plan(self.root)
        payload['source_commit'] = '0' * 40
        (self.root / 'PRIMARY_PLAN.json').write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            driver.plan(self.root)

        self.root = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260815_120003'
        )
        payload = driver.plan(self.root)
        incumbent = payload['incumbents']['cifar100']
        incumbent['metrics']['AA_final'] = -1.0
        compact = {
            key: value for key, value in incumbent.items()
            if key != 'record_sha256'
        }
        incumbent['record_sha256'] = canonical_sha256(compact)
        payload['jobs']['cifar100']['incumbent_record_sha256'] = \
            incumbent['record_sha256']
        (self.root / 'PRIMARY_PLAN.json').write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            driver.plan(self.root)

    def test_ablation_plan_requires_intact_primary_gate_and_stays_separate(self):
        driver.plan(self.root)
        with self.assertRaises(ValueError):
            driver.plan_ablations(self.root)
        report = {'status': 'GATE_SUCCESS'}
        (self.root / 'PRIMARY_GATE.json').write_text(json.dumps(report))
        gate = {
            'status': 'GATE_SUCCESS',
            'primary_plan_sha256': driver.file_sha256(
                self.root / 'PRIMARY_PLAN.json'
            ),
            'gate_report_sha256': driver.file_sha256(
                self.root / 'PRIMARY_GATE.json'
            ),
        }
        (self.root / 'GATE_SUCCESS').write_text(json.dumps(gate))
        (self.root / 'EXECUTION_SUCCESS').write_text(json.dumps({
            **gate, 'status': 'EXECUTION_SUCCESS',
        }))
        with mock.patch.object(
            driver, '_terminal_evidence',
            return_value=('GATE_SUCCESS', report),
        ):
            payload = driver.plan_ablations(self.root)
        self.assertEqual(payload['kind'], 'adaptive_ablation_plan')
        self.assertEqual(len(payload['jobs']), 6)
        self.assertEqual({job['ablation'] for job in payload['jobs'].values()},
                         set(driver.ABLATIONS))
        self.assertTrue((self.root / 'ablations' / 'ABLATION_PLAN.json').is_file())
        self.assertFalse((self.root / 'primary' / 'ABLATION_PLAN.json').exists())

    def test_ablation_plan_rejects_forged_or_failed_primary_marker(self):
        driver.plan(self.root)
        for marker in ('GATE_FAILED', 'GATE_SUCCESS'):
            path = self.root / marker
            path.write_text(json.dumps({
                'status': marker,
                'primary_plan_sha256': '0' * 64,
            }))
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                driver.plan_ablations(self.root)
            path.unlink()

    def test_existing_ablation_plan_is_strictly_revalidated(self):
        driver.plan(self.root)
        report = {'status': 'GATE_SUCCESS'}
        (self.root / 'PRIMARY_GATE.json').write_text(json.dumps(report))
        gate = {
            'status': 'GATE_SUCCESS',
            'primary_plan_sha256': driver.file_sha256(
                self.root / 'PRIMARY_PLAN.json'
            ),
            'gate_report_sha256': driver.file_sha256(
                self.root / 'PRIMARY_GATE.json'
            ),
        }
        (self.root / 'GATE_SUCCESS').write_text(json.dumps(gate))
        (self.root / 'EXECUTION_SUCCESS').write_text(json.dumps({
            **gate, 'status': 'EXECUTION_SUCCESS',
        }))
        with mock.patch.object(
            driver, '_terminal_evidence',
            return_value=('GATE_SUCCESS', report),
        ):
            payload = driver.plan_ablations(self.root)
            first = next(iter(payload['jobs'].values()))
            first['command'][-1] = 'forged'
            path = self.root / 'ablations' / 'ABLATION_PLAN.json'
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                driver.plan_ablations(self.root)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260815_130000'
        )
        self.incumbents = {
            dataset: fake_incumbent(dataset) for dataset in DATASETS
        }
        self.patches = [
            mock.patch.object(
                driver, 'incumbent_evidence',
                side_effect=lambda dataset: json.loads(json.dumps(
                    self.incumbents[dataset]
                )),
            ),
            mock.patch.object(
                driver, 'adaptive_source_identity',
                return_value={
                    'commit': driver.SOURCE_COMMIT,
                    'sha256': {'adaptive_consolidation_audit.py': '1' * 64},
                },
            ),
            mock.patch.object(
                driver, '_read_only_deferred_dataset', return_value=object(),
            ),
            mock.patch.object(
                driver, 'evaluate_deferred_cil_trajectory',
                return_value={'cl_metrics': {
                    metric: 0.5 for metric in METRICS
                }},
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        driver.plan(self.root)

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temporary.cleanup()

    def make_run(self, dataset='cifar100', metrics=None, ablation=None,
                 smoke=False):
        if smoke and not (self.root / 'SMOKE_PLAN.json').exists():
            driver.plan_smoke(self.root)
        phase = ('smoke' if smoke else
                 ('primary' if ablation is None else f'ablations/{ablation}'))
        job = self.root / phase / dataset
        output = job / 'outputs'
        run = output / f'adaptive_{dataset}_seed42_20260815_130001'
        output.mkdir(parents=True, exist_ok=True)
        command = driver.build_command(
            dataset, '0', output, smoke=smoke, ablation=ablation,
        )
        plan_path = job / 'planned_protocol.json'
        plan_payload = driver.bound_job_plan(
            self.root, dataset, '0', smoke=smoke, ablation=ablation,
        )
        launch = {
            'kind': 'adaptive_launch_started',
            'spec': plan_payload['spec'],
            'command': command,
            'planned_protocol_sha256': driver.file_sha256(plan_path),
        }
        (job / 'launch_started.json').write_text(json.dumps(launch))
        values = metrics or {metric: 0.5 for metric in METRICS}
        config = command_flags(command)
        config = {key[2:]: driver.config_value(key, value)
                  for key, value in config.items()}
        config['output_dir'] = str(run)
        freeze = {
            'status': 'ADAPTIVE_STATE_FROZEN',
            'audit_spec': {'checkpoint': 'adaptive_final.pt'},
            'source': {'source_commit': driver.SOURCE_COMMIT},
            'result': {
                'candidate_hashes': {
                    'pre': '1' * 64, 'full': '2' * 64, 'bias': '3' * 64,
                },
                'gate': {
                    'gate_rule': driver.ABLATIONS.get(
                        ablation, 'class_balanced'
                    ),
                    'is_primary': ablation is None,
                    'g': 0.5,
                    'tolerance': 1e-12,
                    'max_iterations': 80,
                },
            },
            'checkpoint': {'path': 'adaptive_final.pt', 'sha256': 'a' * 64},
            'snapshots': [{
                'path': 'adaptive_snapshots/event_0_CIL.pt',
                'event_idx': 0,
                'task_id': 0,
                'introduced_classes': [0, 1],
                'seen_task_classes': {'0': [0, 1]},
            }],
            'data_flow': {
                'candidates_frozen_before_validation': True,
                'validation_before_freeze': False,
                'test_before_install': False,
                'test_used_for_diagnostics': False,
                'solver_input': 'class_balanced_validation_nll',
                'audit_prefix_record_count': 1,
                'audit_prefix_sha256': '9' * 64,
            },
            'diagnostics': {
                'source_splits': {
                    'replay': 'persistent_training_replay',
                    'validation': 'frozen_training_validation',
                    'test_used': False,
                },
                'solver_input': 'class_balanced_validation_nll',
            },
            'replay': {'count': 20, 'ordered_sample_ids': ['x'],
                       'by_class': {'0': ['x']}, 'sha256': 'b' * 64},
            'validation': {'manifest': {
                'sha256': driver.EXPECTED_VALIDATION_HASHES[dataset],
            }},
        }
        files = {
            'config.json': config,
            'results.json': {'cl_metrics': values},
            'ADAPTIVE_STATE_FROZEN.json': freeze,
            'data_flow_audit.jsonl': {'split': 'validation'},
        }
        run.mkdir()
        for name, payload in files.items():
            path = run / name
            if name.endswith('.jsonl'):
                path.write_text(json.dumps(payload) + '\n')
            else:
                path.write_text(json.dumps(payload))
        (run / 'adaptive_final.pt').write_bytes(b'checkpoint')
        return job, run, command, freeze

    def test_real_install_record_projects_standard_smoke_schema(self):
        job, run, _, freeze = self.make_run(smoke=True)
        smoke_validation_hash = 'c' * 64
        freeze['validation']['manifest']['sha256'] = smoke_validation_hash
        (run / 'ADAPTIVE_STATE_FROZEN.json').write_text(json.dumps(freeze))
        with mock.patch.object(
                driver, 'audit_adaptive_checkpoint', return_value=freeze), \
                mock.patch.object(driver, '_run_resume_probe', return_value=True):
            self.assertTrue(driver._install_record(
                self.root, 'cifar100', smoke=True,
            ))
        record = json.loads((job / 'record.json').read_text())
        required = {
            'branch_isolation', 'validation_after_freeze', 'exact_reload',
            'test_after_install', 'adaptive_audit', 'resume',
        }
        self.assertLessEqual(required, set(record['checks']))
        self.assertTrue(all(record['checks'][name] for name in required))
        self.assertEqual(record['validation_hash'], smoke_validation_hash)

    def make_primary_matrix(self, value=0.5):
        return {
            dataset: self.make_run(
                dataset, metrics={metric: value for metric in METRICS},
            )
            for dataset in DATASETS
        }

    @staticmethod
    def stored_task7_audit(run, _audit_spec):
        return json.loads((Path(run) / 'ADAPTIVE_STATE_FROZEN.json').read_text())

    @staticmethod
    def tree_snapshot(root):
        return {
            str(path.relative_to(root)): (
                path.lstat().st_ino,
                path.lstat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in root.rglob('*')
        }

    def test_completed_run_recomputes_full_task7_audit_without_mutating_run(self):
        _, run, command, freeze = self.make_run()
        before = self.tree_snapshot(run)
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ) as audit:
            record = driver.audit_completed_run(
                'cifar100', command, run, self.root,
            )
        self.assertEqual(before, self.tree_snapshot(run))
        audit.assert_called_once_with(run, freeze['audit_spec'])
        self.assertTrue(record['passed'])
        self.assertEqual(record['metrics'], {metric: 0.5 for metric in METRICS})
        self.assertTrue(all(record['checks'].values()))
        self.assertEqual(set(record['paths']), {
            'config', 'results', 'adaptive_freeze', 'adaptive_checkpoint',
            'data_flow_audit',
        })
        self.assertEqual(set(record['paths']), set(record['sha256']))

    def test_real_metrics_tracker_schema_extracts_only_three_gate_metrics(self):
        tracker = MetricsTracker()
        tracker.record_task_accuracies(
            'event_0_CIL', {'task_0': 0.6}, 0.6,
            per_task_taskil={'task_0': 0.7},
        )
        produced = tracker.to_dict()['cl_metrics']
        self.assertGreater(set(produced), set(METRICS))
        _, run, command, freeze = self.make_run(metrics=produced)
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ), mock.patch.object(
            driver, 'evaluate_deferred_cil_trajectory',
            return_value={'cl_metrics': produced},
        ):
            record = driver.audit_completed_run(
                'cifar100', command, run, self.root,
            )
        self.assertEqual(record['metrics'], {
            'AA_final': 0.6, 'BWT': 0.0, 'AA_final_taskil': 0.7,
        })

    def test_results_metrics_must_match_post_freeze_checkpoint_evaluation(self):
        authoritative = {
            'AA_final': 0.5, 'BWT': 0.5, 'AA_final_taskil': 0.5,
        }
        _, run, command, freeze = self.make_run(metrics=authoritative)
        dataset = object()
        deferred = {'cl_metrics': {
            **authoritative, 'AA_cil': 0.5, 'AA_trajectory': [],
        }}
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ), mock.patch.object(
            driver, '_read_only_deferred_dataset',
            return_value=dataset, create=True,
        ) as build_dataset, mock.patch.object(
            driver, 'evaluate_deferred_cil_trajectory',
            return_value=deferred, create=True,
        ) as evaluate:
            record = driver.audit_completed_run(
                'cifar100', command, run, self.root,
            )
            self.assertEqual(record['metrics'], authoritative)

            results_path = run / 'results.json'
            results = json.loads(results_path.read_text())
            results['cl_metrics'] = {
                **results['cl_metrics'],
                'AA_final': 0.9, 'BWT': 0.9, 'AA_final_taskil': 0.9,
            }
            results_path.write_text(json.dumps(results))
            before = self.tree_snapshot(run)
            with self.assertRaises(ValueError):
                driver.audit_completed_run(
                    'cifar100', command, run, self.root,
                )
            self.assertEqual(before, self.tree_snapshot(run))

        self.assertEqual(build_dataset.call_count, 2)
        self.assertEqual(evaluate.call_count, 2)
        snapshots, checkpoint, actual_dataset, task_classes, _ = \
            evaluate.call_args.args
        self.assertEqual(snapshots, [
            run / 'adaptive_snapshots/event_0_CIL.pt'
        ])
        self.assertEqual(checkpoint, run / 'adaptive_final.pt')
        self.assertIs(actual_dataset, dataset)
        self.assertEqual(task_classes, {0: [0, 1]})

    def test_terminal_rejects_deleted_record_after_success(self):
        self.make_primary_matrix()
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint',
            side_effect=self.stored_task7_audit,
        ):
            for dataset in DATASETS:
                self.assertTrue(driver._install_record(self.root, dataset))
            self.assertEqual(driver.summarize(self.root)[0], 0)
            (self.root / 'primary/isolet/record.json').unlink()
            with self.assertRaises(ValueError):
                driver.summarize(self.root)

    def test_terminal_rejects_tampered_record_after_success(self):
        self.make_primary_matrix()
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint',
            side_effect=self.stored_task7_audit,
        ):
            for dataset in DATASETS:
                self.assertTrue(driver._install_record(self.root, dataset))
            self.assertEqual(driver.summarize(self.root)[0], 0)
            path = self.root / 'primary/cifar100/record.json'
            record = json.loads(path.read_text())
            record['metrics']['AA_final'] = 1.0
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                driver.summarize(self.root)

    def test_terminal_rejects_tampered_checkpoint_after_success(self):
        runs = self.make_primary_matrix()
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint',
            side_effect=self.stored_task7_audit,
        ):
            for dataset in DATASETS:
                self.assertTrue(driver._install_record(self.root, dataset))
            self.assertEqual(driver.summarize(self.root)[0], 0)
            runs['cifar100'][1].joinpath(
                'adaptive_final.pt'
            ).write_bytes(b'tampered checkpoint')
            with self.assertRaises(ValueError):
                driver.summarize(self.root)

    def test_gate_loader_reaudits_and_rejects_a_tampered_driver_record(self):
        _, _, _, freeze = self.make_run()
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ):
            self.assertTrue(driver._install_record(self.root, 'cifar100'))
            path = self.root / 'primary/cifar100/record.json'
            record = json.loads(path.read_text())
            record['metrics']['AA_final'] = 1.0
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                driver._strict_record(self.root, 'cifar100')

    def test_audit_rejects_retroactive_or_changed_plan(self):
        job, run, command, freeze = self.make_run()
        later = run.stat().st_mtime_ns + 2_000_000_000
        os.utime(job / 'planned_protocol.json', ns=(later, later))
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ), self.assertRaises(ValueError):
            driver.audit_completed_run('cifar100', command, run, self.root)

    def test_exact_exp_name_results_dir_and_run_name_are_command_bound(self):
        _, run, command, freeze = self.make_run()
        config = json.loads((run / 'config.json').read_text())
        config['exp_name'] = 'copied_other_run'
        (run / 'config.json').write_text(json.dumps(config))
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ), self.assertRaises(ValueError):
            driver.audit_completed_run('cifar100', command, run, self.root)

    def test_real_task7_data_flow_prefix_is_required_and_accepted(self):
        _, run, command, freeze = self.make_run()
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ):
            self.assertTrue(driver.audit_completed_run(
                'cifar100', command, run, self.root,
            )['passed'])
        freeze['data_flow']['audit_prefix_sha256'] = 'short'
        (run / 'ADAPTIVE_STATE_FROZEN.json').write_text(json.dumps(freeze))
        with mock.patch.object(
            driver, 'audit_adaptive_checkpoint', return_value=freeze,
        ), self.assertRaises(ValueError):
            driver.audit_completed_run('cifar100', command, run, self.root)

    def test_audit_rejects_any_adaptive_checkpoint_or_privacy_mismatch(self):
        _, run, command, freeze = self.make_run()
        variants = []
        wrong_audit = json.loads(json.dumps(freeze))
        wrong_audit['checkpoint']['sha256'] = '0' * 64
        variants.append(('task7_reaudit', wrong_audit))
        wrong_data_flow = json.loads(json.dumps(freeze))
        wrong_data_flow['data_flow']['test_before_install'] = True
        variants.append(('data_flow', wrong_data_flow))
        wrong_privacy = json.loads(json.dumps(freeze))
        wrong_privacy['diagnostics']['source_splits']['test_used'] = True
        variants.append(('privacy', wrong_privacy))
        wrong_gate = json.loads(json.dumps(freeze))
        wrong_gate['result']['gate']['is_primary'] = False
        variants.append(('primary_gate', wrong_gate))
        for label, evidence in variants:
            with self.subTest(label=label):
                stored = freeze if label == 'task7_reaudit' else evidence
                (run / 'ADAPTIVE_STATE_FROZEN.json').write_text(json.dumps(stored))
                returned = evidence
                with mock.patch.object(
                    driver, 'audit_adaptive_checkpoint', return_value=returned,
                ), self.assertRaises(ValueError):
                    driver.audit_completed_run('cifar100', command, run, self.root)
        self.assertFalse((run.parent.parent / 'record.json').exists())

    def test_run_job_writes_plan_and_launch_before_subprocess_and_never_adopts(self):
        captured = {}

        def fake_run(command, **kwargs):
            job = self.root / 'primary' / 'cifar100'
            captured['planned'] = (job / 'planned_protocol.json').is_file()
            captured['launched'] = (job / 'launch_started.json').is_file()
            return subprocess.CompletedProcess(command, 7)

        with mock.patch.object(driver.subprocess, 'run', side_effect=fake_run):
            self.assertEqual(driver.run_job(
                self.root, 'cifar100', 42, '0', smoke=False,
            ), 7)
        self.assertEqual(captured, {'planned': True, 'launched': True})
        self.assertTrue((self.root / 'primary/cifar100/FAILED.json').is_file())
        self.assertFalse((self.root / 'primary/cifar100/SUCCESS').exists())

        other = self.root / 'primary/isolet/outputs/already_complete'
        other.mkdir(parents=True)
        (other / 'results.json').write_text('{}')
        with mock.patch.object(driver.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                driver.run_job(self.root, 'isolet', 42, '0')
        process.assert_not_called()
        self.assertFalse((self.root / 'primary/isolet/planned_protocol.json').exists())


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260815_140000'
        )
        def fixture_record(root, dataset):
            path = root / 'primary' / dataset / 'record.json'
            success = path.parent / 'SUCCESS'
            if not path.is_file() or not success.is_file():
                raise ValueError(f'missing primary record for {dataset}')
            return json.loads(path.read_text())

        self.patches = [
            mock.patch.object(
                driver, 'incumbent_evidence',
                side_effect=lambda dataset: fake_incumbent(dataset),
            ),
            mock.patch.object(
                driver, 'adaptive_source_identity',
                return_value={
                    'commit': driver.SOURCE_COMMIT,
                    'sha256': {'adaptive_consolidation_audit.py': '1' * 64},
                },
            ),
            mock.patch.object(
                driver, '_strict_record', side_effect=fixture_record,
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        self.plan = driver.plan(self.root)

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temporary.cleanup()

    def write_records(self, value=0.49):
        for dataset in DATASETS:
            root = self.root / 'primary' / dataset
            root.mkdir(parents=True, exist_ok=True)
            record = {
                'spec': f'{dataset}:adaptive:42',
                'dataset': dataset,
                'seed': 42,
                'ablation': None,
                'smoke': False,
                'validation_hash': driver.EXPECTED_VALIDATION_HASHES[dataset],
                'metrics': {metric: value for metric in METRICS},
                'checks': {'full_adaptive_audit': True,
                           'checkpoint_audit': True, 'privacy_audit': True},
                'passed': True,
            }
            (root / 'record.json').write_text(json.dumps(record))
            (root / 'SUCCESS').write_text('')

    def clear_terminal_evidence(self):
        for name in (
            'EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
            'PRIMARY_GATE.json',
        ):
            path = self.root / name
            if path.exists():
                path.unlink()

    def write_self_consistent_terminal(self, report, status='GATE_SUCCESS'):
        report_path = self.root / 'PRIMARY_GATE.json'
        report_path.write_text(json.dumps(report))
        common = {
            'primary_plan_sha256': driver.file_sha256(
                self.root / 'PRIMARY_PLAN.json'
            ),
            'gate_report_sha256': driver.file_sha256(report_path),
        }
        (self.root / 'EXECUTION_SUCCESS').write_text(json.dumps({
            'status': 'EXECUTION_SUCCESS', **common,
        }))
        (self.root / status).write_text(json.dumps({
            'status': status, **common,
        }))

    def test_nine_constraints_are_independent_and_inclusive(self):
        self.write_records(value=0.49)
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 0)
        self.assertTrue(all(
            check['passed'] for check in report['constraints'].values()
        ))
        self.assertEqual(len(report['constraints']), 9)

        for dataset in DATASETS:
            for metric in METRICS:
                with self.subTest(dataset=dataset, metric=metric):
                    self.clear_terminal_evidence()
                    self.write_records(value=0.49)
                    path = self.root / 'primary' / dataset / 'record.json'
                    record = json.loads(path.read_text())
                    record['metrics'][metric] = float(Decimal('0.49') - Decimal('0.000000000001'))
                    path.write_text(json.dumps(record))
                    code, report = driver.summarize(self.root)
                    self.assertEqual(code, 3)
                    key = f'{dataset}:{metric}'
                    self.assertFalse(report['constraints'][key]['passed'])
                    self.assertEqual(
                        [name for name, item in report['constraints'].items()
                         if not item['passed']], [key],
                    )

    def test_execution_and_scientific_markers_are_distinct_terminal_evidence(self):
        self.write_records(value=0.48)
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 3)
        self.assertTrue((self.root / 'EXECUTION_SUCCESS').is_file())
        self.assertTrue((self.root / 'GATE_FAILED').is_file())
        self.assertFalse((self.root / 'GATE_SUCCESS').exists())
        self.assertEqual(report['status'], 'GATE_FAILED')

        for name in ('EXECUTION_SUCCESS', 'GATE_FAILED', 'PRIMARY_GATE.json'):
            (self.root / name).unlink()
        self.write_records(value=0.5)
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 0)
        self.assertTrue((self.root / 'EXECUTION_SUCCESS').is_file())
        self.assertTrue((self.root / 'GATE_SUCCESS').is_file())
        self.assertFalse((self.root / 'GATE_FAILED').exists())
        marker = json.loads((self.root / 'GATE_SUCCESS').read_text())
        self.assertEqual(marker['primary_plan_sha256'],
                         driver.file_sha256(self.root / 'PRIMARY_PLAN.json'))
        self.assertEqual(marker['gate_report_sha256'],
                         driver.file_sha256(self.root / 'PRIMARY_GATE.json'))

    def test_incomplete_or_invalid_primary_records_never_write_markers(self):
        self.write_records()
        (self.root / 'primary/upmc_food101/record.json').unlink()
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'INCOMPLETE')
        for marker in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED'):
            self.assertFalse((self.root / marker).exists())

    def test_incoherent_terminal_evidence_is_never_trusted(self):
        self.write_records(value=0.5)
        (self.root / 'PRIMARY_GATE.json').write_text(json.dumps({
            'status': 'GATE_FAILED',
        }))
        (self.root / 'GATE_SUCCESS').write_text(json.dumps({
            'status': 'GATE_SUCCESS',
        }))
        with self.assertRaises(ValueError):
            driver.summarize(self.root)

    def test_self_consistent_success_without_current_records_is_rejected(self):
        constraints = {}
        for dataset in DATASETS:
            for metric in METRICS:
                constraints[f'{dataset}:{metric}'] = {
                    'dataset': dataset, 'metric': metric,
                    'actual': 0.5, 'incumbent': 0.5,
                    'floor': '0.49', 'passed': True,
                }
        self.write_self_consistent_terminal({
            'schema_version': 1,
            'status': 'GATE_SUCCESS',
            'primary_plan_sha256': driver.file_sha256(
                self.root / 'PRIMARY_PLAN.json'
            ),
            'records': {
                dataset: {
                    'path': str(
                        self.root / 'primary' / dataset / 'record.json'
                    ),
                    'sha256': '0' * 64,
                }
                for dataset in DATASETS
            },
            'constraints': constraints,
            'ablation_records_excluded': True,
        })
        with self.assertRaises(ValueError):
            driver.summarize(self.root)

    def test_self_consistent_report_with_extra_schema_is_rejected(self):
        self.write_records(value=0.5)
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 0)
        report['unreviewed_extra'] = True
        self.write_self_consistent_terminal(report)
        with self.assertRaises(ValueError):
            driver.summarize(self.root)

    def test_ablation_records_can_never_replace_or_change_primary_gate(self):
        self.write_records(value=0.48)
        for ablation in driver.ABLATIONS:
            for dataset in DATASETS:
                root = self.root / 'ablations' / ablation / dataset
                root.mkdir(parents=True, exist_ok=True)
                (root / 'record.json').write_text(json.dumps({
                    'spec': f'{dataset}:{ablation}:42',
                    'dataset': dataset, 'seed': 42, 'ablation': ablation,
                    'metrics': {metric: 1.0 for metric in METRICS},
                    'checks': {'all': True}, 'passed': True,
                }))
        code, report = driver.summarize(self.root)
        self.assertEqual(code, 3)
        self.assertEqual(report['status'], 'GATE_FAILED')
        self.assertTrue(all(
            item['actual'] == 0.48 for item in report['constraints'].values()
        ))


class SyntheticSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / (
            'adaptive_dual_branch_validation_seed42_20260816_120000'
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_runtime_normalized_list_options_match_frozen_command(self):
        self.assertEqual(
            driver.config_value('--unlearn_after_tasks', '999'), [999],
        )
        self.assertEqual(
            driver.config_value('--unlearn_classes', '0,1;2'), [[0, 1], [2]],
        )
        self.assertEqual(
            driver.config_value('--party_widths', '16,16,16,16'),
            [16, 16, 16, 16],
        )

    def test_tiny_smoke_command_is_two_task_synthetic_image_only(self):
        builder = getattr(driver, 'build_tiny_synthetic_smoke_command', None)
        self.assertIsNotNone(builder, 'controlled Tiny smoke command is missing')
        command = builder(self.root, 'cuda:1')
        flags = command_flags(command)
        fixture = self.root / 'synthetic_tiny_fixture'
        output = self.root / 'smoke' / 'synthetic_tinyimagenet' / 'outputs'
        self.assertEqual(Path(command[0]), driver.PYTHON)
        self.assertEqual(Path(command[1]), driver.WORKTREE / 'main.py')
        self.assertEqual(flags['--data'], 'tinyimagenet')
        self.assertEqual(Path(flags['--data_path']), fixture)
        self.assertEqual(flags['--num_classes'], '4')
        self.assertEqual(flags['--num_tasks'], '2')
        self.assertEqual(flags['--custom_tasks'], '0,1|2,3')
        self.assertEqual(flags['--classes_per_task'], '2')
        self.assertEqual(flags['--num_parties'], '4')
        self.assertEqual(flags['--party_widths'], '16,16,16,16')
        self.assertEqual(flags['--model_type'], 'resnet18')
        self.assertEqual(flags['--aggregation'], 'sum')
        self.assertEqual(flags['--epochs_per_task'], '1')
        self.assertEqual(flags['--num_workers'], '2')
        self.assertEqual(flags['--lambda_validation_per_class'], '50')
        self.assertGreaterEqual(
            52 - int(flags['--lambda_validation_per_class']), 2,
            'synthetic Tiny must leave multiple training samples per class',
        )
        self.assertEqual(flags['--lambda_validation_split_seed'], '20260813')
        self.assertEqual(flags['--head_consolidation_mode'],
                         'adaptive_dual_branch')
        self.assertEqual(flags['--save_task_checkpoints'], '3')
        self.assertEqual(Path(flags['--results_dir']), output)
        joined = '\n'.join(command)
        self.assertNotIn('/home/chase/Yangxx/VF-CL/data/tiny-imagenet-200', joined)
        self.assertNotIn('adaptive_tinyimagenet_heldout', joined)

    def test_tiny_loader_accepts_the_four_class_smoke_fixture_contract(self):
        train = SimpleNamespace(
            classes=[f'n{index:08d}' for index in range(4)],
            targets=[], samples=[],
        )
        validation = SimpleNamespace(
            classes=list(train.classes), targets=[], samples=[],
        )
        dataset = object.__new__(driver.VFLDataset)
        args = SimpleNamespace(
            data_path='/synthetic-only', num_classes=4,
            lambda_validation_enabled=0, bic_enabled=0,
        )
        with mock.patch(
                'data_utils.datasets.ImageFolder',
                side_effect=[train, validation]):
            dataset._init_tinyimagenet(args)
        self.assertIs(dataset.trainset, train)
        self.assertIs(dataset.testset, validation)

    def test_tiny_fixture_is_local_complete_idempotent_and_tamper_evident(self):
        self.root.mkdir()
        manifest = driver.prepare_tiny_smoke_fixture(self.root)
        fixture = self.root / 'synthetic_tiny_fixture' / 'tiny-imagenet-200'
        self.assertEqual(manifest['classes'], 4)
        self.assertEqual(manifest['tasks'], [[0, 1], [2, 3]])
        self.assertEqual(len(manifest['relative_paths']), 212)
        self.assertEqual(len(set(manifest['sha256'].values())), 212)
        self.assertEqual(len(list((fixture / 'train').iterdir())), 4)
        self.assertEqual(len(list((fixture / 'val').iterdir())), 4)
        self.assertEqual(driver.prepare_tiny_smoke_fixture(self.root), manifest)
        self.assertNotIn(
            '/home/chase/Yangxx/VF-CL/data/tiny-imagenet-200',
            json.dumps(manifest, sort_keys=True),
        )
        extra = fixture / 'train' / 'n00000000' / 'images' / 'extra.png'
        extra.write_bytes(b'untracked image')
        with self.assertRaisesRegex(ValueError, 'paths changed'):
            driver.prepare_tiny_smoke_fixture(self.root)
        extra.unlink()
        sample = fixture / 'train' / 'n00000000' / 'images' / 'n00000000_000.JPEG'
        sample.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            driver.prepare_tiny_smoke_fixture(self.root)

    def test_smoke_plan_is_immutable_and_separate_from_scientific_gate(self):
        planner = getattr(driver, 'plan_smoke', None)
        self.assertIsNotNone(planner, 'smoke plan entry is missing')
        self.root.mkdir()
        with mock.patch.object(driver, 'incumbent_evidence', side_effect=AssertionError(
                'synthetic smoke must not inspect scientific incumbents')), \
                mock.patch.object(driver, 'adaptive_source_identity', return_value={
            'commit': driver.SOURCE_COMMIT,
            'sha256': {'source.py': 'a' * 64},
        }):
            payload = planner(self.root)
            self.assertEqual(payload, planner(self.root))
        self.assertEqual(payload['kind'], 'adaptive_synthetic_smoke_plan')
        self.assertEqual(list(payload['jobs']), [
            'cifar100', 'isolet', 'upmc_food101', 'synthetic_tinyimagenet',
        ])
        self.assertTrue(all(job['smoke'] is True
                            for job in payload['jobs'].values()))
        self.assertNotIn('metrics', payload)
        self.assertNotIn('tolerance', payload)
        self.assertFalse((self.root / 'PRIMARY_PLAN.json').exists())
        self.assertFalse(any((self.root / name).exists() for name in (
            'EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
            'PRIMARY_GATE.json',
        )))

    def test_smoke_audit_writes_only_non_scientific_success(self):
        audit = getattr(driver, 'audit_smoke', None)
        self.assertIsNotNone(audit, 'smoke-only audit entry is missing')
        self.root.mkdir()
        (self.root / 'SMOKE_PLAN.json').write_text('{}\n')
        jobs = {
            name: {'smoke': True, 'command_sha256': str(index) * 64}
            for index, name in enumerate((
                'cifar100', 'isolet', 'upmc_food101',
                'synthetic_tinyimagenet',
            ), start=1)
        }
        plan = {
            'kind': 'adaptive_synthetic_smoke_plan', 'jobs': jobs,
            'source_commit': driver.SOURCE_COMMIT,
        }
        records = {name: {
            'dataset': name, 'smoke': True, 'passed': True,
            'checks': {
                'branch_isolation': True,
                'validation_after_freeze': True,
                'exact_reload': True,
                'test_after_install': True,
                'adaptive_audit': True,
                'resume': True,
            },
        } for name in jobs}
        with mock.patch.object(driver, '_validate_smoke_plan', return_value=plan), \
                mock.patch.object(driver, '_smoke_records', return_value=records):
            report = audit(self.root)
        self.assertEqual(report['status'], 'SMOKE_EXECUTION_SUCCESS')
        self.assertEqual(set(report['records']), set(jobs))
        self.assertTrue((self.root / 'SMOKE_EXECUTION_SUCCESS').is_file())
        self.assertTrue((self.root / 'SMOKE_AUDIT.json').is_file())
        for marker in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
                       'PRIMARY_GATE.json'):
            self.assertFalse((self.root / marker).exists())

    def test_standard_smoke_checks_are_derived_from_real_task7_evidence(self):
        projector = getattr(driver, '_standard_smoke_checks', None)
        self.assertIsNotNone(projector, 'real standard smoke check mapping is missing')
        freeze = {
            'status': 'ADAPTIVE_STATE_FROZEN',
            'result': {'candidate_hashes': {
                'pre': 'a' * 64, 'full': 'b' * 64, 'bias': 'c' * 64,
            }},
            'data_flow': {
                'candidates_frozen_before_validation': True,
                'validation_before_freeze': False,
                'test_before_install': False,
                'test_used_for_diagnostics': False,
            },
        }
        self.assertEqual(projector(freeze, freeze, True), {
            'branch_isolation': True,
            'validation_after_freeze': True,
            'exact_reload': True,
            'test_after_install': True,
            'adaptive_audit': True,
            'resume': True,
        })
        self.assertFalse(projector(freeze, freeze, False)['resume'])

    def test_resume_probe_uses_runner_checkpoint_admission_in_fresh_process(self):
        probe = getattr(driver, '_run_resume_probe', None)
        self.assertIsNotNone(probe, 'checkpoint resume probe is missing')
        completed = SimpleNamespace(returncode=0, stdout='RESUME_PROBE_SUCCESS\n')
        with mock.patch.object(driver.subprocess, 'run', return_value=completed) as run:
            self.assertTrue(probe(Path('/run/fixture')))
        command = run.call_args.args[0]
        self.assertEqual(Path(command[0]), driver.PYTHON)
        self.assertEqual(command[1], '-c')
        self.assertIn('_strict_resume_probe_main', command[2])
        self.assertEqual(command[3], '/run/fixture')
        failed = SimpleNamespace(returncode=7, stdout='bad resume')
        with mock.patch.object(driver.subprocess, 'run', return_value=failed):
            with self.assertRaisesRegex(ValueError, 'resume probe failed'):
                probe(Path('/run/fixture'))

    def test_resume_probe_preserves_the_frozen_checkpoint_device(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps({
                'device': 'cuda:7', 'cl_method': 'proto_evolve',
            }))
            checkpoints = run / 'checkpoints'
            checkpoints.mkdir()
            (checkpoints / 'event_0_CIL.pt').write_bytes(b'checkpoint-zero')
            manager = mock.Mock()
            manager.get_timeline.return_value = [
                {'type': 'CIL', 'task_id': 0},
                {'type': 'CIL', 'task_id': 1},
            ]
            order = []
            dataset_roots = []

            def build(args):
                self.assertEqual(args.party_col_ranges, [(0, 2), (2, 4)])
                self.assertEqual(Path(args.output_dir), dataset_roots[0])
                self.assertEqual(Path(args.resume_run_dir), run)
                order.append('models')
                return [], object()

            def dataset(args):
                order.append('dataset')
                output = Path(args.output_dir)
                self.assertNotEqual(output, run)
                output.mkdir(parents=True, exist_ok=True)
                (output / 'validation_manifest.json').write_text('{}\n')
                dataset_roots.append(output)
                args.party_col_ranges = [(0, 2), (2, 4)]
                return object()

            def load(args, *unused):
                self.assertEqual(args.device, 'cuda:7')
                output = Path(args.output_dir)
                self.assertEqual(output, dataset_roots[0])
                self.assertEqual(
                    (output / 'checkpoints' / 'event_0_CIL.pt').read_bytes(),
                    b'checkpoint-zero',
                )
                (output / 'recovered_snapshot.pt').write_bytes(b'recovered')
                return 2, {0: [0, 1], 1: [2, 3]}, []

            with mock.patch('data_utils.VFLDataset', side_effect=dataset), \
                    mock.patch('models.build_models', side_effect=build), \
                    mock.patch(
                        'vfl_trainer.VFLTrainer', return_value=SimpleNamespace(),
                    ), \
                    mock.patch('data_utils.TaskManager', return_value=manager), \
                    mock.patch('cl_methods.get_cl_method', return_value=object()), \
                    mock.patch('runner._load_resume_checkpoint', side_effect=load), \
                    contextlib.redirect_stdout(io.StringIO()):
                driver._strict_resume_probe_main(run)
            self.assertEqual(order, ['dataset', 'models'])
            self.assertEqual(len(dataset_roots), 1)
            self.assertFalse(dataset_roots[0].exists())
            self.assertFalse((run / 'validation_manifest.json').exists())
            self.assertFalse((run / 'recovered_snapshot.pt').exists())

    def test_resume_probe_checkpoint_copy_rejects_symlinks(self):
        copier = getattr(driver, '_copy_resume_checkpoints', None)
        self.assertIsNotNone(copier, 'safe resume checkpoint copier is missing')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            target = root / 'target'
            source.mkdir()
            outside = root / 'outside.pt'
            outside.write_bytes(b'outside')
            (source / 'event_0_CIL.pt').symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'symlink|regular'):
                copier(source, target)
            self.assertFalse(target.exists())


class CliAndLauncherTest(unittest.TestCase):
    launcher = Path(__file__).with_name('run_adaptive_dual_branch_validation.sh')

    def setUp(self):
        patcher = mock.patch.dict(
            os.environ, {'VFCL_PYTHON': str(driver.PYTHON)},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cli_supports_only_the_listed_actions(self):
        parser = driver.build_parser()
        accepted = [
            ['check'], ['plan', '--root', '/tmp/root'],
            ['plan-ablations', '--root', '/tmp/root'],
            ['run-job', '--root', '/tmp/root', '--dataset', 'cifar100',
             '--seed', '42', '--device', '0'],
            ['run-job', '--root', '/tmp/root', '--dataset', 'isolet',
             '--seed', '42', '--device', '1', '--smoke'],
            ['audit', '--root', '/tmp/root'],
            ['summarize', '--root', '/tmp/root'],
            ['plan-smoke', '--root', '/tmp/root'],
            ['run-tiny-smoke', '--root', '/tmp/root', '--device', '1'],
            ['audit-smoke', '--root', '/tmp/root'],
        ]
        for arguments in accepted:
            with self.subTest(arguments=arguments):
                parser.parse_args(arguments)
        for action in ('jobs', 'audit-reuse', 'main', 'smoke', 'restart'):
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([action])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([
                'run-job', '--root', '/tmp/root', '--dataset', 'cifar100',
                '--seed', '43', '--device', '0',
            ])

    def test_launcher_exists_is_executable_and_has_valid_bash_syntax(self):
        self.assertTrue(self.launcher.is_file())
        self.assertTrue(self.launcher.stat().st_mode & stat.S_IXUSR)
        completed = subprocess.run(
            ['bash', '-n', str(self.launcher)], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_contract_is_exact_and_forbids_bypass_restart_paths(self):
        text = self.launcher.read_text()
        required = [
            'W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)',
            '${VFCL_PYTHON:?VFCL_PYTHON must name the reviewed Python interpreter}',
            'PY=$(realpath -e -- "$VFCL_PYTHON")',
            'git -C "$W" rev-parse --git-common-dir',
            '[[ "$(basename -- "$COMMON")" == .git ]]',
            'REPO=$(dirname -- "$COMMON")',
            'RESULTS_BASE=${RESULTS_BASE:-$REPO/results}',
            'pathlib.Path(sys.executable).resolve()!=pathlib.Path(sys.argv[1]).resolve()',
            driver.SOURCE_COMMIT,
            'adaptive_dual_branch_validation_seed42_',
            'git status --porcelain', 'git rev-parse HEAD',
            'PRIMARY_PLAN.json', 'LAUNCHER_TOKEN', '/proc/$pid/stat',
            '/proc/$pid/cmdline', 'start_ticks', 'pgid',
            'setsid', 'kill -- -', 'wait', 'worker_0', 'worker_1',
            'run-job', 'summarize', 'EXECUTION_SUCCESS',
            '--smoke', 'plan-smoke', 'run-tiny-smoke', 'audit-smoke',
            'SMOKE_EXECUTION_SUCCESS',
        ]
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        self.assertNotIn('/home/chase', text)
        self.assertLess(text.index('pathlib.Path(sys.executable)'),
                        text.index('ROOT=${1:-'))
        self.assertEqual(text.count('worker_loop 0'), 1)
        self.assertEqual(text.count('worker_loop 1'), 1)
        forbidden = [
            'main.py', 'tmux', 'nohup', 'git reset', 'git checkout',
            'git clean', '--seed 43', '--seed 44', 'restart_job',
        ]
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, text)

    def test_launcher_source_mode_root_and_identity_helpers_fail_closed(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
RESULTS_BASE="$tmp"
ROOT="$tmp/adaptive_dual_branch_validation_seed42_20260815_150000"
prepare_root
safe_target "$ROOT/logs/launcher.log"
mkdir "$tmp/outside"
ln -s "$tmp/outside" "$ROOT/bad"
if safe_target "$ROOT/bad/file"; then exit 31; fi
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_ignores_external_smoke_mode_at_top_level(self):
        command = f'''set -eu
export SMOKE_MODE=1
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
[[ "$SMOKE_MODE" == 0 ]]
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_freezes_deterministic_seed42_environment(self):
        command = f'''set -eu
export CUBLAS_WORKSPACE_CONFIG=wrong
export OMP_NUM_THREADS=9
export MKL_NUM_THREADS=8
export PYTHONHASHSEED=7
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
[[ "$CUBLAS_WORKSPACE_CONFIG" == :4096:8 ]]
[[ "$OMP_NUM_THREADS" == 1 ]]
[[ "$MKL_NUM_THREADS" == 1 ]]
[[ "$PYTHONHASHSEED" == 42 ]]
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gpu_claims_are_atomic_owner_bound_and_can_fallback(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
declare -F claim_gpu >/dev/null
declare -F release_gpu >/dev/null
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
RESULTS_BASE="$tmp"
ROOT="$tmp/adaptive_dual_branch_validation_seed42_20260816_120000"
mkdir -p "$ROOT"
LAUNCHER_TOKEN=token
process_identity() {{ printf '1\t2\tbash-test\n'; }}
gpu_available() {{ [[ "$1" == 1 ]]; }}
gpu=$(wait_gpu 0)
[[ "$gpu" == 1 ]]
owner="$ROOT/gpu_claims/gpu_1/owner"
[[ -f "$owner" && ! -L "$owner" ]]
grep -Fx 'token=token' "$owner" >/dev/null
grep -Fx 'role=worker_0:gpu_1' "$owner" >/dev/null
if claim_gpu 1 1; then exit 31; fi
LAUNCHER_TOKEN=wrong
if release_gpu 1 0; then exit 32; fi
[[ -f "$owner" ]]
LAUNCHER_TOKEN=token
release_gpu 1 0
[[ ! -e "$ROOT/gpu_claims/gpu_1" ]]
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_smoke_workers_cover_four_shapes_without_scientific_gate(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ROOT="$tmp/adaptive_dual_branch_validation_seed42_20260816_120000"
mkdir -p "$ROOT"
SMOKE_MODE=1
LAUNCHER_TOKEN=token
claim_job() {{ return 0; }}
release_job() {{ return 0; }}
wait_gpu() {{
  printf '%s\n' "$1" >>"$tmp/wait_gpu"
  [[ "$1" == 0 ]] && printf '1\n' || printf '0\n'
}}
release_gpu() {{ printf '%s:%s\n' "$1" "$2" >>"$tmp/release_gpu"; }}
tracked_driver() {{ printf '%s\n' "$*" >>"$tmp/calls"; }}
worker_loop 0
worker_loop 1
grep -F -- 'run-job --root ' "$tmp/calls" >/dev/null
[[ "$(grep -c -- '--smoke' "$tmp/calls")" == 3 ]]
[[ "$(grep -c -- 'run-tiny-smoke' "$tmp/calls")" == 1 ]]
[[ "$(grep -c -- '--device cuda:0' "$tmp/calls")" == 2 ]]
[[ "$(grep -c -- '--device cuda:1' "$tmp/calls")" == 2 ]]
[[ "$(grep -c -x -- '0' "$tmp/wait_gpu")" == 2 ]]
[[ "$(grep -c -x -- '1' "$tmp/wait_gpu")" == 2 ]]
[[ "$(sort "$tmp/release_gpu")" == $'0:1\n0:1\n1:0\n1:0' ]]
! grep -E 'summarize|GATE_SUCCESS|GATE_FAILED' "$tmp/calls"
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_smoke_main_never_invokes_scientific_plan_or_gate(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ROOT="$tmp/adaptive_dual_branch_validation_seed42_20260816_120000"
mkdir -p "$ROOT"
IMPLEMENTATION_COMMIT=commit
preflight() {{ :; }}
prepare_root() {{ :; }}
atomic_text() {{ :; }}
claim_launcher() {{ :; }}
cleanup() {{ :; }}
start_worker() {{ :; }}
wait_workers() {{ return 0; }}
git() {{ [[ "$1" == rev-parse ]] && printf '%s\n' commit; }}
tracked_driver() {{
  printf '%s\n' "$*" >>"$tmp/calls"
  [[ "$2" != audit-smoke ]] || touch "$ROOT/SMOKE_EXECUTION_SUCCESS"
}}
smoke_main
[[ "$(cat "$tmp/calls")" == $'smoke_plan plan-smoke --root '*$'\nsmoke_audit audit-smoke --root '* ]]
! grep -E 'primary_plan|(^| )plan( |$)|summarize|gate' "$tmp/calls"
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_rejects_symlinked_results_base_itself(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/real"
ln -s "$tmp/real" "$tmp/base"
RESULTS_BASE="$tmp/base"
ROOT="$RESULTS_BASE/adaptive_dual_branch_validation_seed42_20260815_150001"
if safe_target "$ROOT"; then exit 31; fi
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_preflight_rejects_an_untracked_executable_copy(self):
        command = f'''set -eu
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "{self.launcher}" "$tmp/copied.sh"
if (export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1; source "$tmp/copied.sh"); then
  exit 31
fi
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cleanup_always_visits_registered_groups_even_after_success(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
terminate_group() {{ printf '%s\n' "$1" >>"$tmp/groups"; }}
( STOPPING=0; true; cleanup )
[[ "$(wc -l <"$tmp/groups")" == 2 ]]
rm -rf "$tmp"
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_worker1_failure_immediately_terminates_live_worker0_group(self):
        command = f'''set -eu
export ADAPTIVE_LAUNCHER_SOURCE_ONLY=1
source "{self.launcher}"
tmp=$(mktemp -d)
WORKER_0_PID=101
WORKER_1_PID=202
worker_running() {{ [[ "$1" == 101 ]]; }}
wait() {{ [[ "$1" == 202 ]] && return 7; return 0; }}
terminate_group() {{ printf '%s\n' "$1" >>"$tmp/terminated"; }}
sleep() {{ :; }}
status=0
wait_workers || status=$?
[[ "$status" == 7 ]]
[[ "$(<"$tmp/terminated")" == 0 ]]
rm -rf "$tmp"
'''
        completed = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
