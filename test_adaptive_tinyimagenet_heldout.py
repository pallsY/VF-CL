import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import adaptive_tinyimagenet_heldout as heldout


class AdaptiveTinyImageNetHeldoutTest(unittest.TestCase):
    def manifest(self):
        hashes = dict(zip(
            ('paths', 'classes', 'tasks', 'transforms', 'split', 'manifest'),
            (character * 64 for character in '123456'),
        ))
        return {
            'schema_version': 1,
            'dataset_root': str(heldout.DATA_ROOT),
            'archive_sha256': 'a' * 64,
            'paths_sha256': hashes['paths'],
            'class_order_sha256': hashes['classes'],
            'task_manifest_sha256': hashes['tasks'],
            'transforms_sha256': hashes['transforms'],
            'split_sha256': hashes['split'],
            'manifest_sha256': hashes['manifest'],
            'class_order': [f'n{index:08d}' for index in range(200)],
            'task_manifest': [
                [f'n{index:08d}' for index in range(start, start + 20)]
                for start in range(0, 200, 20)
            ],
            'protocol': heldout.frozen_protocol(),
        }

    def gate(self):
        return {
            'status': 'GATE_SUCCESS',
            'root': '/development',
            'marker_sha256': 'd' * 64,
            'report_sha256': 'e' * 64,
            'plan_sha256': 'f' * 64,
        }

    def source(self):
        return {
            'commit': '1' * 40,
            'clean': True,
            'source_sha256': {name: '2' * 64 for name in heldout.SOURCE_FILES},
        }

    def test_static_contract_exact_jobs_and_protocol(self):
        self.assertEqual(heldout.REPO, heldout.development.REPO)
        self.assertEqual(heldout.WORKTREE, heldout.development.WORKTREE)
        self.assertEqual(heldout.PYTHON, heldout.development.PYTHON)
        self.assertEqual(heldout.DATA_ROOT, heldout.REPO / 'data' / 'tiny-imagenet-200')
        self.assertEqual(heldout.PRIMARY_VARIANTS, ('full', 'bias', 'adaptive'))
        self.assertEqual(heldout.PRIMARY_SEED, 42)
        self.assertEqual(heldout.FORMAL_SEEDS, (43, 44))
        self.assertEqual(heldout.ABLATIONS, (
            'no_consolidation', 'fixed_half', 'sample_mean_nll',
        ))
        self.assertEqual(heldout.METRICS, ('AA_final', 'BWT', 'AA_final_taskil'))
        self.assertEqual(heldout.PROTOCOL['validation_split_seed'], 20260813)
        self.assertEqual(heldout.PROTOCOL['solver_tolerance'], 1e-12)
        self.assertEqual(heldout.PROTOCOL['solver_max_iterations'], 80)

    def test_development_gate_requires_real_task8_success(self):
        root = Path('/development')
        with mock.patch.object(
                heldout.development, '_terminal_evidence',
                return_value=('GATE_SUCCESS', {'status': 'GATE_SUCCESS'})), \
                mock.patch.object(heldout, 'file_sha256', side_effect=[
                    'd' * 64, 'e' * 64, 'f' * 64,
                ]):
            evidence = heldout.validate_development_gate(root)
        self.assertEqual(evidence, self.gate())
        with mock.patch.object(
                heldout.development, '_terminal_evidence',
                return_value=('GATE_FAILED', {'status': 'GATE_FAILED'})):
            with self.assertRaisesRegex(ValueError, 'GATE_SUCCESS'):
                heldout.validate_development_gate(root)

    def test_one_way_freeze_is_exclusive_and_revalidates_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = heldout.freeze_payload(
                self.manifest(), self.gate(), self.source()
            )
            path = heldout.install_freeze(root, payload)
            self.assertEqual(path.name, 'HELDOUT_PROTOCOL_FROZEN.json')
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(heldout.install_freeze(root, payload), path)
            changed = dict(payload, source_commit='3' * 40)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                heldout.install_freeze(root, changed)

    def test_missing_freeze_cannot_be_installed_over_existing_evidence(self):
        evidence_paths = (
            'HELDOUT_PRIMARY_PLAN.json',
            'primary/adaptive/planned_protocol.json',
            'primary/adaptive/launch_started.json',
            'primary/adaptive/record.json',
            'primary/adaptive/outputs/run/config.json',
            'HELDOUT_PRIMARY_GATE.json',
            'HELDOUT_GATE_SUCCESS',
        )
        payload = heldout.freeze_payload(
            self.manifest(), self.gate(), self.source()
        )
        for relative in evidence_paths:
            with self.subTest(relative=relative), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                artifact = root / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'existing evidence'):
                    heldout.install_freeze(root, payload)

    def test_recreated_identical_freeze_is_rejected_before_synthetic_run_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = heldout.freeze_payload(
                self.manifest(), self.gate(), self.source()
            )
            freeze_path = heldout.install_freeze(root, payload)
            patches = (
                mock.patch.object(heldout, 'source_identity',
                                  return_value=self.source()),
                mock.patch.object(heldout, 'validate_development_gate',
                                  return_value=self.gate()),
                mock.patch.object(heldout, '_validate_dataset_manifest'),
                mock.patch.object(heldout, 'build_heldout_manifest',
                                  return_value=self.manifest()),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                primary = heldout.plan(root)
                job_root = root / 'primary' / 'adaptive'
                command = [
                    '0' if value == '__DEVICE__' else value
                    for value in primary['primary_jobs']['adaptive']['command']
                ]
                bound = {
                    'schema_version': 1, 'kind': 'heldout_bound_job_plan',
                    'variant': 'adaptive', 'seed': 42, 'device': '0',
                    'command': command,
                    'command_sha256': heldout.payload_sha256(command),
                    'freeze_identity': primary['freeze_identity'],
                    'primary_plan_sha256': heldout.file_sha256(
                        root / 'HELDOUT_PRIMARY_PLAN.json'
                    ),
                }
                heldout._write_exclusive(
                    job_root / 'planned_protocol.json', bound,
                )
                heldout._write_exclusive(job_root / 'launch_started.json', {
                    'kind': 'heldout_launch_started',
                    'variant': 'adaptive', 'seed': 42,
                    'command': bound['command'],
                    'planned_protocol_sha256': heldout.file_sha256(
                        job_root / 'planned_protocol.json'
                    ),
                })
                run = (job_root / 'outputs' /
                       'heldout_tinyimagenet_adaptive_seed42_20260816_120000')
                run.mkdir(parents=True)
                (run / 'config.json').write_text('{}')
                (run / 'results.json').write_text('{}')
                original_hash = heldout.file_sha256(freeze_path)
                original_identity = primary['freeze_identity']
                self.assertEqual(heldout._validate_plan(root), primary)
                import adaptive_consolidation_audit as audit_module

                def consume_two_loaders(
                        snapshot_paths, final_checkpoint, dataset,
                        task_classes, args):
                    dataset.get_test_loader([0])
                    dataset.get_test_loader([1])
                    return {'metrics': {}}

                dataset = mock.Mock()
                with mock.patch.object(
                        audit_module, 'evaluate_deferred_cil_trajectory',
                        side_effect=consume_two_loaders), \
                        mock.patch.object(
                            heldout, '_validate_runtime_freeze',
                            wraps=heldout._validate_runtime_freeze,
                        ) as full_validation, \
                        mock.patch.object(
                            heldout, '_validate_live_freeze',
                            wraps=heldout._validate_live_freeze,
                        ) as cheap_validation:
                    with heldout._runtime_test_guard(
                            root, 'adaptive', 42, original_identity):
                        audit_module.evaluate_deferred_cil_trajectory(
                            [], run / 'adaptive_final.pt', dataset, {},
                            mock.Mock(),
                        )
                self.assertEqual(full_validation.call_count, 1)
                self.assertEqual(cheap_validation.call_count, 3)
                freeze_path.unlink()
                with self.assertRaisesRegex(ValueError, 'existing evidence'):
                    heldout.install_freeze(root, payload)
                heldout._write_exclusive(freeze_path, payload)
                self.assertEqual(heldout.file_sha256(freeze_path), original_hash)
                self.assertNotEqual(
                    heldout.freeze_identity(freeze_path), original_identity,
                )
                with self.assertRaisesRegex(ValueError, 'freeze identity'):
                    heldout.audit_completed_run(
                        root, 'adaptive', 42, run,
                    )
                delegate = mock.Mock(return_value={'metrics': {}})
                dataset = mock.Mock()
                with mock.patch.object(
                        audit_module, 'evaluate_deferred_cil_trajectory',
                        delegate):
                    with heldout._runtime_test_guard(root, 'adaptive', 42):
                        wrapped = audit_module.evaluate_deferred_cil_trajectory
                        with self.assertRaisesRegex(ValueError, 'freeze identity'):
                            wrapped([], run / 'adaptive_final.pt', dataset, {},
                                    mock.Mock())
                delegate.assert_not_called()
                dataset.get_test_loader.assert_not_called()

    def test_freeze_must_predate_plan_launch_and_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_path = root / 'HELDOUT_PROTOCOL_FROZEN.json'
            artifact = root / 'results.json'
            freeze_path.write_text('{}')
            artifact.write_text('{}')
            heldout._validate_freeze_precedes(freeze_path, [artifact])
            future = artifact.stat().st_mtime_ns + 1_000_000_000
            os.utime(freeze_path, ns=(future, future))
            with self.assertRaisesRegex(ValueError, 'predate'):
                heldout._validate_freeze_precedes(freeze_path, [artifact])

    def test_freeze_revalidation_rejects_tamper_and_symlink_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = heldout.freeze_payload(
                self.manifest(), self.gate(), self.source()
            )
            path = heldout.install_freeze(root, payload)
            with mock.patch.object(heldout, '_validate_dataset_manifest'), \
                    mock.patch.object(heldout, 'build_heldout_manifest',
                                      return_value=self.manifest()), \
                    mock.patch.object(heldout, 'source_identity',
                                      return_value=self.source()), \
                    mock.patch.object(heldout, 'validate_development_gate',
                                      return_value=self.gate()):
                self.assertEqual(heldout._load_freeze(root), payload)
                path.chmod(0o644)
                path.write_text(json.dumps(dict(payload, protocol={})))
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    heldout._load_freeze(root)
            real = root / 'real'
            real.mkdir()
            link = root / 'link'
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                heldout._write_exclusive(link / 'forbidden.json', {})

    def test_freeze_rejects_a_dataset_root_other_than_the_runtime_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(self.manifest(), dataset_root=str(root / 'copy'))
            payload = heldout.freeze_payload(manifest, self.gate(), self.source())
            heldout.install_freeze(root, payload)
            with mock.patch.object(heldout, '_validate_dataset_manifest'), \
                    mock.patch.object(heldout, 'build_heldout_manifest',
                                      return_value=manifest), \
                    mock.patch.object(heldout, 'source_identity',
                                      return_value=self.source()), \
                    mock.patch.object(heldout, 'validate_development_gate',
                                      return_value=self.gate()):
                with self.assertRaisesRegex(ValueError, 'runtime data root'):
                    heldout._load_freeze(root)

    def test_retroactive_artifacts_and_invalid_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'results.json'
            artifact.write_text('{}')
            os.utime(artifact, (1, 1))
            plan = root / 'planned_protocol.json'
            launch = root / 'launch_started.json'
            plan.write_text('{}')
            launch.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'retroactive'):
                heldout._reject_retroactive_artifacts(plan, launch, [artifact])
            real = root / 'real_validation'
            real.mkdir()
            nested = real / 'validation_manifest.json'
            nested.write_text('{}')
            link = root / 'validation'
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlinked'):
                heldout._reject_retroactive_artifacts(
                    plan, launch, [link / nested.name]
                )
        for invalid in (True, 1.1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                heldout._metrics({'metrics': {
                    'AA_final': invalid, 'BWT': 0.0, 'AA_final_taskil': 0.0,
                }})

    def test_plan_requires_freeze_and_preregisters_only_seed42_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'freeze'):
                heldout.plan(root)
            payload = heldout.freeze_payload(
                self.manifest(), self.gate(), self.source()
            )
            heldout.install_freeze(root, payload)
            with mock.patch.object(heldout, 'source_identity', return_value=self.source()), \
                    mock.patch.object(heldout, 'validate_development_gate',
                                      return_value=self.gate()), \
                    mock.patch.object(heldout, '_validate_dataset_manifest'), \
                    mock.patch.object(heldout, 'build_heldout_manifest',
                                      return_value=self.manifest()):
                plan = heldout.plan(root)
            self.assertEqual(list(plan['primary_jobs']), list(heldout.PRIMARY_VARIANTS))
            self.assertEqual(
                {job['seed'] for job in plan['primary_jobs'].values()}, {42}
            )
            self.assertEqual(plan['formal_seeds'], [43, 44])
            self.assertEqual(plan['ablations'], list(heldout.ABLATIONS))
            self.assertEqual(len(plan['followup_jobs']), 15)
            self.assertFalse((root / 'FOLLOWUP_AUTHORIZATION.json').exists())

    def test_commands_freeze_exact_tinyimagenet_protocol(self):
        command = heldout.build_command(
            'adaptive', 42, '0', Path('/results'), self.source()['commit']
        )
        self.assertEqual(command[1], '-c')
        self.assertIn('_execute_heldout_main', command[2])
        flags = dict(zip(command[3::2], command[4::2]))
        expected = {
            '--data': 'tinyimagenet', '--num_classes': '200',
            '--num_tasks': '10', '--classes_per_task': '20',
            '--num_parties': '4', '--party_widths': '16,16,16,16',
            '--model_type': 'resnet18', '--aggregation': 'sum',
            '--seed': '42', '--lambda_validation_enabled': '1',
            '--lambda_validation_per_class': '50',
            '--lambda_validation_split_seed': '20260813',
            '--head_consolidation_mode': 'adaptive_dual_branch',
            '--head_gate_rule': 'class_balanced',
        }
        for option, value in expected.items():
            self.assertEqual(flags[option], value)
        self.assertNotIn('test', ' '.join(command).lower())

        for variant, expected_gate in (
                ('full', 1.0), ('bias', 0.0), ('no_consolidation', 0.5)):
            fixed = heldout.build_command(
                variant, 42, '0', Path('/results'), self.source()['commit']
            )
            self.assertEqual(fixed[1], '-c')
            self.assertIn('_execute_heldout_main', fixed[2])
            fixed_flags = dict(zip(fixed[3::2], fixed[4::2]))
            self.assertEqual(
                fixed_flags['--head_consolidation_mode'],
                'adaptive_dual_branch',
            )
            self.assertEqual(heldout.fixed_branch_gate(variant), expected_gate)

    def test_fixed_branch_runtime_records_audited_exact_endpoints(self):
        import adaptive_head_consolidation as head
        import cl_methods.proto_evolve as proto

        full = torch.log_softmax(
            torch.tensor([[3., 1.], [1., 3.]], dtype=torch.float64), dim=1,
        )
        bias = torch.log_softmax(
            torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float64), dim=1,
        )
        labels = torch.tensor([0, 1])
        for variant, expected in (
                ('full', 1.0), ('bias', 0.0), ('no_consolidation', 0.5)):
            with heldout._fixed_branch_runtime(variant):
                gate = proto.solve_global_mixture_weight(
                    full, bias, labels, [0, 1]
                )
                result = head.AdaptiveConsolidationResult(
                    pre_head_sha256='a' * 64,
                    candidate_hashes={
                        'pre': 'a' * 64, 'full': 'b' * 64, 'bias': 'c' * 64,
                    },
                    candidate_configs={
                        'full': head.FULL_BRANCH_CONFIG,
                        'bias': head.BIAS_BRANCH_CONFIG,
                    },
                    gate=gate, validation_manifest={
                        'dataset': 'fixture', 'seed': 1, 'per_class': 1,
                        'by_class': {'0': ['a'], '1': ['b']},
                        'ordered_sample_ids': ['a', 'b'],
                        'sha256': heldout.payload_sha256(['a', 'b']),
                    },
                    ordered_classes=(0, 1), task_id=0,
                    task_boundary='event_0_CIL',
                )
            self.assertEqual(gate['g'], expected)
            self.assertEqual(
                gate['gate_rule'],
                'no_consolidation' if variant == 'no_consolidation'
                else f'fixed_{variant}',
            )
            self.assertFalse(gate['is_primary'])
            self.assertEqual(result.to_dict()['gate'], gate)
            if variant in ('full', 'bias'):
                inactive = 'bias' if variant == 'full' else 'full'
                self.assertEqual(
                    result.to_dict()['candidate_configs'][inactive],
                    {'mode': 'inactive', 'parameters': 0},
                )

    def test_fixed_branch_runtime_never_calls_dual_fit(self):
        import adaptive_head_consolidation as head
        import cl_methods.proto_evolve as proto

        sentinel = object()
        for variant in ('full', 'bias'):
            with self.subTest(variant=variant), mock.patch.object(
                    head, 'fit_fixed_endpoint_candidate',
                    return_value=sentinel, create=True) as fixed_fit, \
                    mock.patch.object(
                        proto, 'fit_adaptive_candidates',
                        side_effect=AssertionError('dual fit must not run'),
                    ):
                with heldout._fixed_branch_runtime(variant):
                    result = proto.fit_adaptive_candidates(
                        'pre', 'replay', 'prototypes', 'tasks', 4, 42, 'cpu',
                    )
                self.assertIs(result, sentinel)
                self.assertEqual(fixed_fit.call_count, 1)
                self.assertEqual(fixed_fit.call_args.args[1], variant)

    def test_run_discovery_and_config_are_bound_to_the_exact_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            job_root = Path(directory)
            output = job_root / 'outputs'
            output.mkdir()
            exp_name = 'heldout_tinyimagenet_adaptive_seed42'
            exact = output / f'{exp_name}_20260816_120000'
            forged = output / f'{exp_name}_20260816_120001_extra'
            for run in (exact, forged):
                run.mkdir()
                (run / 'config.json').write_text('{}')
                (run / 'results.json').write_text('{}')
            self.assertEqual(
                heldout._discover_run(job_root, 'adaptive', 42), exact,
            )
            config = {
                'exp_name': exp_name, 'results_dir': str(output),
                'output_dir': str(exact),
            }
            heldout._validate_run_identity(
                exact, output, 'adaptive', 42, config,
            )
            with self.assertRaisesRegex(ValueError, 'identity'):
                heldout._validate_run_identity(
                    exact, output, 'adaptive', 42,
                    dict(config, output_dir=str(forged)),
                )

    def test_gate_has_three_independent_floors_and_strict_gain(self):
        records = {
            'full': {'metrics': {'AA_final': .70, 'BWT': -.10, 'AA_final_taskil': .80}},
            'bias': {'metrics': {'AA_final': .69, 'BWT': -.08, 'AA_final_taskil': .82}},
            'adaptive': {'metrics': {'AA_final': .691, 'BWT': -.09, 'AA_final_taskil': .821}},
        }
        passed, report = heldout.primary_gate_report(records)
        self.assertTrue(passed)
        self.assertEqual(len(report['constraints']), 3)
        self.assertTrue(report['strict_improvement'])
        boundary = json.loads(json.dumps(records))
        boundary['adaptive']['metrics']['AA_final_taskil'] = .82 + 1e-12
        passed, report = heldout.primary_gate_report(boundary)
        self.assertFalse(passed)
        self.assertFalse(report['strict_improvement'])
        failed = json.loads(json.dumps(records))
        failed['adaptive']['metrics']['BWT'] = -.0900000000001
        self.assertFalse(heldout.primary_gate_report(failed)[0])

    def test_followups_are_illegal_until_intact_success_and_then_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'success'):
                heldout.authorize_followups(root)
            gate = {'status': 'HELDOUT_GATE_SUCCESS', 'strict_improvement': True}
            (root / 'HELDOUT_PRIMARY_GATE.json').write_text(json.dumps(gate))
            (root / 'HELDOUT_GATE_SUCCESS').write_text(json.dumps({
                'status': 'HELDOUT_GATE_SUCCESS',
                'report_sha256': heldout.file_sha256(root / 'HELDOUT_PRIMARY_GATE.json'),
            }))
            with mock.patch.object(heldout, 'validate_terminal_gate', return_value=gate), \
                    mock.patch.object(heldout, '_validate_plan',
                                      return_value={'source_commit': '1' * 40}):
                authorization = heldout.authorize_followups(root)
            self.assertEqual(authorization['formal_seeds'], [43, 44])
            self.assertEqual(authorization['ablations'], list(heldout.ABLATIONS))
            self.assertEqual(authorization['primary_seed'], 42)
            self.assertEqual(len(authorization['jobs']), 15)
            self.assertEqual(
                {job['seed'] for job in authorization['jobs'].values()},
                {42, 43, 44},
            )

    def test_run_job_authorization_is_primary_first_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = heldout.freeze_payload(
                self.manifest(), self.gate(), self.source()
            )
            heldout.install_freeze(root, payload)
            with mock.patch.object(heldout, 'source_identity', return_value=self.source()), \
                    mock.patch.object(heldout, 'validate_development_gate',
                                      return_value=self.gate()), \
                    mock.patch.object(heldout, '_validate_dataset_manifest'), \
                    mock.patch.object(heldout, 'build_heldout_manifest',
                                      return_value=self.manifest()):
                heldout.plan(root)
                with mock.patch.object(heldout, '_run_one', return_value=0) as run:
                    self.assertEqual(heldout.run_job(root, 'adaptive', 42, '0'), 0)
                    run.assert_called_once_with(root, 'adaptive', 42, '0')
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    heldout.run_job(root, 'adaptive', 43, '0')
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    heldout.run_job(root, 'fixed_half', 42, '0')

    def test_summarize_installs_distinct_execution_and_scientific_markers(self):
        records = {
            'full': {'metrics': {'AA_final': .70, 'BWT': -.10, 'AA_final_taskil': .80}},
            'bias': {'metrics': {'AA_final': .69, 'BWT': -.08, 'AA_final_taskil': .82}},
            'adaptive': {'metrics': {'AA_final': .691, 'BWT': -.09, 'AA_final_taskil': .821}},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                heldout, '_strict_record',
                side_effect=lambda root, variant: records[variant]):
            root = Path(directory)
            (root / 'HELDOUT_PRIMARY_PLAN.json').write_text('{}')
            code, report = heldout.summarize(root)
            self.assertEqual(code, 0)
            self.assertEqual(report['status'], 'HELDOUT_GATE_SUCCESS')
            self.assertTrue((root / 'EXECUTION_SUCCESS').is_file())
            self.assertTrue((root / 'HELDOUT_GATE_SUCCESS').is_file())
            self.assertFalse((root / 'HELDOUT_GATE_FAILED').exists())

    def test_cli_exposes_only_bounded_execution_arguments(self):
        parser = heldout.build_parser()
        args = parser.parse_args([
            'run-job', '--root', '/heldout', '--variant', 'adaptive',
            '--seed', '42', '--device', '0',
        ])
        self.assertEqual((args.variant, args.seed, args.device), ('adaptive', 42, '0'))
        with self.assertRaises(SystemExit):
            parser.parse_args([
                'run-job', '--root', '/heldout', '--variant', 'adaptive',
                '--seed', '45', '--device', '0',
            ])

    def test_launcher_is_hardened_and_driver_only(self):
        text = (Path(__file__).parent / 'run_adaptive_tinyimagenet_heldout.sh').read_text()
        for literal in (
            'W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)',
            '${VFCL_PYTHON:?VFCL_PYTHON must name the reviewed Python interpreter}',
            'PY=$(realpath -e -- "$VFCL_PYTHON")',
            'git -C "$W" rev-parse --git-common-dir',
            '[[ "$(basename -- "$COMMON")" == .git ]]',
            'REPO=$(dirname -- "$COMMON")',
            'RESULTS_BASE=${RESULTS_BASE:-$REPO/results}',
            'pathlib.Path(sys.executable).resolve()!=pathlib.Path(sys.argv[1]).resolve()',
            'adaptive_tinyimagenet_heldout_seed42_', 'safe_target()',
            'git status --porcelain', 'git merge-base --is-ancestor',
            'process_identity()', 'claim_matches()', 'terminate_group()',
            'wait_workers()', 'setsid bash', 'driver freeze', 'driver plan',
            'run-job', 'summarize', 'driver authorize-followups',
            'FOLLOWUP_PHASE', 'seed 43', 'seed 44',
            'no_consolidation', 'fixed_half', 'sample_mean_nll',
        ):
            self.assertIn(literal, text)
        self.assertNotIn('/home/chase', text)
        self.assertLess(text.index('pathlib.Path(sys.executable)'),
                        text.index('ROOT=${1:-'))
        self.assertNotIn('main.py', text)
        self.assertNotIn('prepare_tinyimagenet.py', text)
        self.assertNotIn('${DATA_ROOT:-', text)
        self.assertLess(
            text.index('FOLLOWUP_PHASE=1'), text.index('FOLLOWUP_PHASE=2')
        )


if __name__ == '__main__':
    unittest.main()
