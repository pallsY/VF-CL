"""Contract tests for the unified head-consolidation factorial matrix."""
import contextlib
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import config
import unified_head_consolidation_factorial as factorial


def as_flags(command):
    assert len(command) % 2 == 0
    return dict(zip(command[2::2], command[3::2]))


class MatrixContractTest(unittest.TestCase):
    def test_matrix_names_cells_and_seed_are_frozen(self):
        self.assertEqual(factorial.DATASET_NAMES, ('cifar100', 'isolet', 'upmc_food101'))

    def test_validation_hashes_are_exactly_frozen(self):
        self.assertEqual(factorial.EXPECTED_VALIDATION_HASHES, {
            'cifar100': '0aa4729ade65021ce774c757516584c4d2a2ce70d2879b045db47dbceae917fa',
            'isolet': '487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3',
            'upmc_food101': 'b812e99da856d94ee5b6eea28eadaf9924f52e355d44941c42da4028c0895e06',
        })
        self.assertEqual(tuple(factorial.CELLS), ('A', 'B', 'C', 'D'))
        specs = factorial.all_specs()
        self.assertEqual(len(specs), 12)
        self.assertTrue(all(spec.endswith(':42') for spec in specs))

    def test_validation_manifest_labels_are_exactly_frozen(self):
        self.assertEqual(factorial.EXPECTED_VALIDATION_LABELS, {
            'cifar100': 'cifar100-train',
            'isolet': 'isolet_vfl.npz-train',
            'upmc_food101': 'upmc_food101_vfl.npz-train',
        })

    def test_cells_are_frozen(self):
        self.assertEqual(factorial.CELLS, {
            'A': ('full_classifier', 0.01, 500),
            'B': ('task_class_bias', 0.03, 600),
            'C': ('full_classifier', 0.03, 600),
            'D': ('task_class_bias', 0.01, 500),
        })

    def test_worker_partitions_are_complete_and_disjoint(self):
        first = set(factorial.jobs(0, 2))
        second = set(factorial.jobs(1, 2))
        self.assertFalse(first & second)
        self.assertEqual(first | second, set(factorial.all_specs()))

    def test_parse_spec_rejects_non_contract_seeds(self):
        for seed in (41, 43, 44):
            with self.assertRaises(ValueError):
                factorial.parse_spec(f'cifar100:A:{seed}')


class CommandContractTest(unittest.TestCase):
    def command(self, dataset, cell, smoke=False):
        return factorial.build_command(
            f'{dataset}:{cell}:42', 'cuda:0',
            Path('/tmp/factorial-contract') / dataset / cell, smoke=smoke,
        )

    def test_commands_keep_required_invariants(self):
        for spec in factorial.all_specs():
            command = self.command(*factorial.parse_spec(spec)[:2])
            flags = as_flags(command)
            self.assertEqual(command[1], str(factorial.WORKTREE / 'main.py'))
            self.assertEqual(flags['--seed'], '42')
            self.assertEqual(flags['--head_consolidation_schedule'], 'final')
            self.assertEqual(flags['--head_consolidation_samples_per_class'], '20')
            self.assertEqual(flags['--lambda_validation_enabled'], '1')
            self.assertEqual(flags['--deterministic'], '1')

    def test_c_and_d_only_change_registered_factors_and_outputs(self):
        for dataset in factorial.DATASET_NAMES:
            first = as_flags(self.command(dataset, 'C'))
            second = as_flags(self.command(dataset, 'D'))
            self.assertEqual(set(first), set(second))
            self.assertEqual({key for key in set(first) | set(second) if first[key] != second[key]}, {
                '--head_consolidation_mode', '--head_consolidation_lr',
                '--head_consolidation_steps', '--results_dir', '--exp_name',
            })

    def test_dataset_fields_are_constant_across_cells(self):
        frozen = (
            '--data', '--model_type', '--aggregation', '--epochs_per_task',
            '--batch_size', '--optimizer', '--lr', '--bottom_lr_scale',
            '--weight_decay', '--num_tasks', '--num_parties', '--task_ce_mode',
            '--lambda_validation_per_class', '--lambda_validation_split_seed',
        )
        for dataset in factorial.DATASET_NAMES:
            flags = [as_flags(self.command(dataset, cell)) for cell in factorial.CELLS]
            for flag in frozen:
                self.assertEqual({command[flag] for command in flags}, {flags[0][flag]})

    def test_registered_factors_match_every_cell(self):
        for dataset in factorial.DATASET_NAMES:
            for cell, expected in factorial.CELLS.items():
                flags = as_flags(self.command(dataset, cell))
                self.assertEqual((
                    flags['--head_consolidation_mode'],
                    float(flags['--head_consolidation_lr']),
                    int(flags['--head_consolidation_steps']),
                ), expected)

    def test_smoke_commands_keep_contract_with_one_epoch_and_two_tasks(self):
        for spec in factorial.all_specs():
            dataset, cell, _ = factorial.parse_spec(spec)
            command = self.command(dataset, cell, smoke=True)
            flags = as_flags(command)
            self.assertEqual(flags['--epochs_per_task'], '1')
            self.assertEqual(flags['--num_tasks'], '2')
            self.assertEqual(flags['--replay_mode'], 'prototype')
            self.assertEqual(flags['--seed'], '42')
            self.assertEqual(flags['--lambda_validation_enabled'], '1')
            self.assertEqual((
                flags['--head_consolidation_mode'],
                float(flags['--head_consolidation_lr']),
                int(flags['--head_consolidation_steps']),
            ), factorial.CELLS[cell])
            self.assertEqual(command[1], str(factorial.WORKTREE / 'main.py'))

    def test_resume_run_dir_is_added_only_when_requested(self):
        for dataset in factorial.DATASET_NAMES:
            command = factorial.build_command(
                f'{dataset}:A:42', 'cuda:0', '/tmp/factorial-contract',
                smoke=True, resume_run_dir='/tmp/resume',
            )
            self.assertEqual(as_flags(command)['--resume_run_dir'], '/tmp/resume')
            self.assertNotIn('--resume_run_dir', self.command(dataset, 'A', smoke=True))


class ReuseAuditTest(unittest.TestCase):
    def test_deployment_paths_use_git_common_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / 'VF-CL'
            common = root / '.git'
            common.mkdir(parents=True)
            worktree = temporary / 'worktrees' / 'adaptive'
            worktree.mkdir(parents=True)
            python = temporary / 'envs' / 'mlz' / 'bin' / 'python'
            python.parent.mkdir(parents=True)
            python.touch()
            actual = factorial.deployment_paths(
                module_file=worktree / 'unified_head_consolidation_factorial.py',
                executable=python,
                common_dir=common,
            )
        self.assertEqual(actual, (root, worktree, python))

    def test_local_reference_storage_preserves_canonical_declared_identity(self):
        self.patchers[0].stop()
        relative = factorial.REFERENCE_RELATIVE_RUNS[('cifar100', 'B')]
        self.assertEqual(
            factorial.REFERENCE_RUNS[('cifar100', 'B')],
            factorial.ROOT / relative,
        )
        self.assertEqual(
            factorial.REFERENCE_DECLARED_RUNS[('cifar100', 'B')],
            Path('/home/chase/Yangxx/VF-CL') / relative,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.matrix_root = self.root / 'matrix'
        self.data = self.root / 'data'
        self.data.mkdir()
        self.npz = self.data / 'isolet_vfl.npz'
        self.metadata = self.data / 'isolet_vfl.metadata.json'
        self.npz.write_bytes(b'fixture npz')
        self.metadata.write_text('{"fixture": true}', encoding='utf-8')
        self.run = self.root / 'reference'
        self._write_reference()
        self.provenance = self.root / 'isolet_MATCHED_PROTOCOL.json'
        self._write_json(self.provenance, {
            'dataset': 'isolet', 'seed': 42,
            'selection_source': 'training-validation',
            'selection_test_used': False,
            'code_commit': factorial.SOURCE_COMMIT,
        })
        self.patchers = [
            mock.patch.object(factorial, 'REFERENCE_RUNS', {('isolet', 'A'): self.run}),
            mock.patch.object(factorial, 'REFERENCE_PROVENANCE', {'isolet': self.provenance}),
            mock.patch.object(factorial, 'EXPECTED_DATA_ARTIFACTS', {
                'isolet': {
                    self.npz: self._hash(self.npz),
                    self.metadata: self._hash(self.metadata),
                },
            }),
            mock.patch.object(factorial, 'EXPECTED_VALIDATION_HASHES', {
                'isolet': 'fixture-validation-hash',
            }),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    def _expected_config(self):
        dry_root = Path(tempfile.mkdtemp(dir=self.root))
        command = factorial.build_command('isolet:A:42', 'cuda:0', dry_root)
        with mock.patch.object(sys, 'argv', command[1:]), \
                contextlib.redirect_stdout(io.StringIO()):
            return vars(config.get_config())

    def _write_reference(self):
        config_payload = self._expected_config()
        self._write_json(self.run / 'config.json', config_payload)
        self._write_json(self.run / 'results.json', {
            'selection_audit': {
                'passed': True,
                'evaluation_source': 'vector-train-validation',
                'test_used_for_selection': False,
            },
            'cl_metrics': {
                'AA_final': 0.7, 'BWT': -0.1, 'AA_final_taskil': 0.9,
            },
        })
        self._write_json(self.run / 'validation' / 'validation_manifest.json', {
            'dataset': 'isolet_vfl.npz-train',
            'sha256': 'fixture-validation-hash',
            'seed': 20260809, 'per_class': 40,
        })
        (self.run / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.run / 'checkpoints' / 'event_12_CIL.pt').write_bytes(b'checkpoint')
        self._write_json(self.run / 'head_consolidation' / 'event_12_CIL.json', {
            'mode': 'full_classifier', 'lr': 0.01, 'steps': 500,
            'replay_selection': 'normalized_feature_herding',
            'samples_per_class': 20, 'schedule': 'final',
            'source': 'balanced_current_encoder_raw_replay',
            'persistent_embedding_count': 0,
            'persistent_raw_example_count': 520, 'class_count': 26,
            'test_used': False, 'validation_used': False,
        })

    def _load(self, relative):
        return json.loads((self.run / relative).read_text(encoding='utf-8'))

    def _audit(self, matrix_root=None):
        return factorial.audit_reference(
            'isolet', 'A', self.matrix_root if matrix_root is None else matrix_root,
        )

    def test_valid_reference_writes_passed_atomic_record(self):
        record = self._audit()
        written = json.loads(
            (self.matrix_root / 'reuse_audit' / 'isolet_A.json').read_text(encoding='utf-8')
        )
        self.assertTrue(record['passed'])
        self.assertEqual(record, written)
        self.assertEqual(record['spec'], 'isolet:A:42')
        self.assertEqual(record['metrics'], {
            'AA_final': 0.7, 'BWT': -0.1, 'AA_final_taskil': 0.9,
        })
        self.assertTrue(all(record['source_sha256'].values()))

    def test_wrong_commit_is_rejected(self):
        payload = json.loads(self.provenance.read_text(encoding='utf-8'))
        payload['code_commit'] = '430be2f'
        self._write_json(self.provenance, payload)
        record = self._audit()
        self.assertFalse(record['checks']['source_commit'])
        self.assertFalse(record['passed'])

    def test_missing_commit_evidence_is_rejected(self):
        payload = json.loads(self.provenance.read_text(encoding='utf-8'))
        payload.pop('code_commit')
        self._write_json(self.provenance, payload)
        record = self._audit()
        self.assertIsNone(record['actual_source_commit'])
        self.assertFalse(record['checks']['source_commit'])

    def test_dataset_artifact_hash_mismatch_is_rejected(self):
        self.npz.write_bytes(b'changed')
        record = self._audit()
        self.assertFalse(record['checks']['dataset_artifacts'])
        self.assertFalse(record['passed'])

    def test_validation_logical_hash_mismatch_is_rejected(self):
        manifest = self._load('validation/validation_manifest.json')
        manifest['sha256'] = 'wrong'
        self._write_json(self.run / 'validation' / 'validation_manifest.json', manifest)
        record = self._audit()
        self.assertFalse(record['checks']['validation_logical_hash'])

    def test_non_authoritative_validation_label_is_rejected(self):
        manifest = self._load('validation/validation_manifest.json')
        manifest['dataset'] = 'isolet'
        self._write_json(self.run / 'validation/validation_manifest.json', manifest)
        record = self._audit()
        self.assertFalse(record['checks']['validation_manifest_label'])
        self.assertFalse(record['passed'])

    def test_wrong_validation_seed_or_per_class_is_rejected(self):
        for key, value in (('seed', 1), ('per_class', 1)):
            with self.subTest(key=key):
                self._write_reference()
                manifest = self._load('validation/validation_manifest.json')
                manifest[key] = value
                self._write_json(
                    self.run / 'validation/validation_manifest.json', manifest,
                )
                record = self._audit()
                self.assertFalse(record['checks']['validation_manifest_contract'])
                self.assertFalse(record['passed'])

    def test_forged_training_validation_source_is_rejected(self):
        results = self._load('results.json')
        results['selection_audit']['evaluation_source'] = 'forged-train-validation'
        self._write_json(self.run / 'results.json', results)
        record = self._audit()
        self.assertFalse(record['checks']['selection_audit'])
        self.assertFalse(record['passed'])

    def test_non_factor_effective_config_mismatch_is_rejected(self):
        config_payload = self._load('config.json')
        config_payload['batch_size'] += 1
        self._write_json(self.run / 'config.json', config_payload)
        record = self._audit()
        self.assertFalse(record['checks']['effective_config'])
        self.assertIn('batch_size', record['effective_config_mismatches'])

    def test_legacy_defaults_are_compatible_only_when_expected_exactly(self):
        defaults = {
            'formal_deferred_evaluation': False,
            'fedprotip_tip_threshold': 0.775,
            'fedprotip_max_batches': 20,
        }
        expected = self._load('config.json')
        for key, value in defaults.items():
            with self.subTest(key=key):
                missing = dict(expected)
                missing.pop(key)
                self.assertEqual(
                    factorial._reference_config_mismatches(expected, missing), {},
                )
                wrong = (
                    not value if type(value) is bool
                    else value + 1 if type(value) in (int, float)
                    else 'wrong'
                )
                present_wrong = dict(missing)
                present_wrong[key] = wrong
                self.assertIn(
                    key,
                    factorial._reference_config_mismatches(expected, present_wrong),
                )
                wrong_expected = dict(expected)
                wrong_expected[key] = wrong
                self.assertIn(
                    key,
                    factorial._reference_config_mismatches(wrong_expected, missing),
                )

    def test_legacy_defaults_require_exact_value_types(self):
        expected = self._load('config.json')
        self.assertIs(type(expected['formal_deferred_evaluation']), bool)
        self.assertIs(type(expected['fedprotip_tip_threshold']), float)
        self.assertIs(type(expected['fedprotip_max_batches']), int)

        missing = dict(expected)
        missing.pop('formal_deferred_evaluation')
        self.assertEqual(
            factorial._reference_config_mismatches(expected, missing), {},
        )

        present_zero = dict(expected)
        present_zero['formal_deferred_evaluation'] = 0
        self.assertIn(
            'formal_deferred_evaluation',
            factorial._reference_config_mismatches(expected, present_zero),
        )

        expected_zero = dict(expected)
        expected_zero['formal_deferred_evaluation'] = 0
        self.assertIn(
            'formal_deferred_evaluation',
            factorial._reference_config_mismatches(expected_zero, missing),
        )

        present_float = dict(expected)
        present_float['fedprotip_max_batches'] = 20.0
        self.assertIn(
            'fedprotip_max_batches',
            factorial._reference_config_mismatches(expected, present_float),
        )

        expected_float = dict(expected)
        expected_float['fedprotip_max_batches'] = 20.0
        missing_batches = dict(expected)
        missing_batches.pop('fedprotip_max_batches')
        self.assertIn(
            'fedprotip_max_batches',
            factorial._reference_config_mismatches(
                expected_float, missing_batches,
            ),
        )

    def test_missing_post_source_defaults_are_legacy_compatible(self):
        config_payload = self._load('config.json')
        for key in factorial._POST_SOURCE_REFERENCE_DEFAULTS:
            config_payload.pop(key)
        self._write_json(self.run / 'config.json', config_payload)
        record = self._audit()
        self.assertTrue(record['checks']['effective_config'])
        self.assertTrue(record['passed'])

    def test_present_wrong_post_source_defaults_are_rejected(self):
        for key, value in factorial._POST_SOURCE_REFERENCE_DEFAULTS.items():
            with self.subTest(key=key):
                self._write_reference()
                config_payload = self._load('config.json')
                config_payload[key] = (
                    not value if type(value) is bool
                    else value + 1 if type(value) in (int, float)
                    else 'wrong'
                )
                self._write_json(self.run / 'config.json', config_payload)
                record = self._audit(self.root / f'matrix-wrong-{key}')
                self.assertFalse(record['checks']['effective_config'])
                self.assertIn(key, record['effective_config_mismatches'])

    def test_each_factor_mismatch_is_rejected(self):
        for key, wrong in (
            ('head_consolidation_mode', 'task_class_bias'),
            ('head_consolidation_lr', 0.03),
            ('head_consolidation_steps', 600),
        ):
            with self.subTest(key=key):
                self._write_reference()
                config_payload = self._load('config.json')
                config_payload[key] = wrong
                self._write_json(self.run / 'config.json', config_payload)
                record = self._audit(self.root / f'matrix-{key}')
                self.assertFalse(record['checks']['factors'])

    def test_selection_and_fit_leakage_are_rejected(self):
        cases = (
            ('results.json', ('selection_audit', 'passed'), False),
            ('results.json', ('selection_audit', 'test_used_for_selection'), True),
            ('results.json', ('test_used_for_fit',), True),
            ('head_consolidation/event_12_CIL.json', ('test_used',), True),
            ('head_consolidation/event_12_CIL.json', ('validation_used',), True),
        )
        for relative, keys, value in cases:
            with self.subTest(relative=relative, keys=keys):
                self._write_reference()
                payload = self._load(relative)
                target = payload
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self._write_json(self.run / relative, payload)
                record = self._audit(self.root / ('matrix-leak-' + keys[-1]))
                self.assertFalse(record['passed'])

    def test_rejected_ab_are_appended_after_all_required_cd_jobs(self):
        outcomes = {('cifar100', cell): False for cell in ('A', 'B')}
        outcomes.update({(dataset, cell): True for dataset in ('isolet', 'upmc_food101') for cell in ('A', 'B')})
        with mock.patch.object(
            factorial, 'audit_reference',
            side_effect=lambda dataset, cell, root: {
                'spec': f'{dataset}:{cell}:42', 'passed': outcomes[(dataset, cell)],
            },
        ):
            payload = factorial.audit_reuse(self.matrix_root)
        self.assertEqual(payload['required_jobs'], [
            'cifar100:C:42', 'isolet:C:42', 'upmc_food101:C:42',
            'cifar100:D:42', 'isolet:D:42', 'upmc_food101:D:42',
            'cifar100:A:42', 'cifar100:B:42',
        ])
        self.assertEqual(payload['reused_jobs'], [
            'isolet:A:42', 'isolet:B:42',
            'upmc_food101:A:42', 'upmc_food101:B:42',
        ])

    def test_required_jobs_are_deterministic_and_unique(self):
        with mock.patch.object(
            factorial, 'audit_reference',
            side_effect=lambda dataset, cell, root: {
                'spec': f'{dataset}:{cell}:42', 'passed': False,
            },
        ):
            first = factorial.audit_reuse(self.root / 'matrix-first')
            second = factorial.audit_reuse(self.root / 'matrix-second')
        self.assertEqual(first, second)
        self.assertEqual(len(first['required_jobs']), len(set(first['required_jobs'])))

    def test_reference_tree_content_and_mtimes_are_unchanged(self):
        before = {
            path.relative_to(self.run): (self._hash(path), path.stat().st_mtime_ns)
            for path in self.run.rglob('*') if path.is_file()
        }
        self._audit()
        after = {
            path.relative_to(self.run): (self._hash(path), path.stat().st_mtime_ns)
            for path in self.run.rglob('*') if path.is_file()
        }
        self.assertEqual(before, after)

    def test_only_missing_mode_is_normalized_to_full_classifier(self):
        config_payload = self._load('config.json')
        config_payload.pop('head_consolidation_mode')
        self._write_json(self.run / 'config.json', config_payload)
        self.assertTrue(self._audit(self.root / 'matrix-missing-mode')['passed'])

        self._write_reference()
        config_payload = self._load('config.json')
        config_payload.pop('head_consolidation_lr')
        self._write_json(self.run / 'config.json', config_payload)
        record = self._audit(self.root / 'matrix-missing-lr')
        self.assertFalse(record['checks']['factors'])

    def test_replay_count_contract_matches_completed_run_audit(self):
        cases = (
            ('boolean-raw', True, 26, False),
            ('boolean-class', 520, True, False),
            ('negative', -1, 26, False),
            ('wrong-class', 500, 25, False),
            ('overflow', 521, 26, False),
            ('zero-boundary', 0, 26, True),
            ('upper-boundary', 520, 26, True),
        )
        for name, raw_count, class_count, expected in cases:
            with self.subTest(name=name):
                self._write_reference()
                head_path = self.run / 'head_consolidation/event_12_CIL.json'
                head = self._load('head_consolidation/event_12_CIL.json')
                head['persistent_raw_example_count'] = raw_count
                head['class_count'] = class_count
                self._write_json(head_path, head)
                record = self._audit(self.root / f'matrix-replay-{name}')
                self.assertIs(record['checks']['replay_selection'], expected)
                self.assertIs(record['passed'], expected)

    def test_reuse_runtime_write_targets_reject_directory_and_file_symlinks(self):
        for name in ('reuse_audit', 'required_jobs.json'):
            with self.subTest(name=name):
                matrix = self.root / ('matrix-symlink-' + name.replace('.', '-'))
                matrix.mkdir()
                outside = self.root / ('outside-' + name.replace('.', '-'))
                if name == 'reuse_audit':
                    outside.mkdir()
                    (matrix / name).symlink_to(outside, target_is_directory=True)
                else:
                    outside.write_text('unchanged', encoding='utf-8')
                    (matrix / name).symlink_to(outside)
                before = list(outside.iterdir()) if outside.is_dir() else outside.read_bytes()
                with self.assertRaises(ValueError):
                    factorial._reuse_write_targets(matrix)
                after = list(outside.iterdir()) if outside.is_dir() else outside.read_bytes()
                self.assertEqual(after, before)


class RunJobTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.matrix_root = self.root / 'matrix'
        self.spec = 'isolet:A:42'
        self.data = self.root / 'data'
        self.data.mkdir()
        self.npz = self.data / 'isolet_vfl.npz'
        self.metadata = self.data / 'isolet_vfl.metadata.json'
        self.npz.write_bytes(b'fixture npz')
        self.metadata.write_text('{"fixture": true}', encoding='utf-8')
        self.patchers = [
            mock.patch.object(factorial, 'EXPECTED_DATA_ARTIFACTS', {
                'isolet': {
                    self.npz: self._hash(self.npz),
                    self.metadata: self._hash(self.metadata),
                },
            }),
            mock.patch.object(factorial, 'EXPECTED_VALIDATION_HASHES', {
                'isolet': 'fixture-validation-hash',
            }),
            mock.patch.object(
                factorial, '_implementation_commit', return_value='fixture-implementation-commit',
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    def _root(self, smoke=False):
        return factorial.job_root(self.matrix_root, self.spec, smoke=smoke)

    def _command(self, smoke=False, resume=None):
        return factorial.build_command(
            self.spec, 'cuda:0', self._root(smoke), smoke=smoke,
            resume_run_dir=resume,
        )

    def _config(self, command):
        command = list(command)
        if '--resume_run_dir' in command:
            index = command.index('--resume_run_dir')
            del command[index:index + 2]
        command[command.index('--results_dir') + 1] = tempfile.mkdtemp(dir=self.root)
        with mock.patch.object(sys, 'argv', command[1:]), \
                contextlib.redirect_stdout(io.StringIO()):
            return vars(config.get_config())

    def _write_run(self, smoke=False, command=None, run=None):
        expected_tasks = 2 if smoke else 13
        run = run or self._root(smoke) / 'run_fixture'
        command = command or self._command(smoke)
        self._write_json(run / 'config.json', self._config(command))
        trajectory = [
            {'step': f'event_{index}_CIL', 'AA': 0.8}
            for index in range(expected_tasks)
        ]
        self._write_json(run / 'results.json', {
            'selection_audit': {
                'passed': True,
                'evaluation_source': 'vector-train-validation',
                'test_used_for_selection': False,
            },
            'test_used_for_fit': False,
            'cl_metrics': {
                'AA_final': 0.7, 'BWT': -0.1, 'AA_final_taskil': 0.9,
                'AA_trajectory_taskil': trajectory,
            },
        })
        self._write_json(run / 'validation' / 'validation_manifest.json', {
            'dataset': 'isolet_vfl.npz-train',
            'sha256': 'fixture-validation-hash',
            'seed': 20260809, 'per_class': 40,
        })
        final = expected_tasks - 1
        checkpoint = run / 'checkpoints' / f'event_{final}_CIL.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'checkpoint')
        self._write_json(run / 'head_consolidation' / f'event_{final}_CIL.json', {
            'mode': 'full_classifier', 'lr': 0.01, 'steps': 500,
            'replay_selection': 'normalized_feature_herding',
            'samples_per_class': 20, 'schedule': 'final',
            'source': 'balanced_current_encoder_raw_replay',
            'persistent_embedding_count': 0,
            'persistent_raw_example_count': 80 if smoke else 520,
            'class_count': 4 if smoke else 26,
            'task_boundary': f'event_{final}_CIL', 'task_id': final,
            'test_used': False, 'validation_used': False,
        })
        (run / 'data_flow_audit.jsonl').write_text(
            json.dumps({'split': 'train', 'test_used_for_fit': False}) + '\n',
            encoding='utf-8',
        )
        return run

    def _audit(self, run, smoke=False, command=None):
        return factorial.audit_completed_run(
            self.spec, command or self._command(smoke), run,
            self.matrix_root, smoke=smoke,
        )

    def _plan_payload(self, command=None, smoke=False):
        command = list(command or self._command(smoke))
        flags = as_flags(command)
        return {
            'spec': self.spec, 'dataset': 'isolet', 'cell': 'A', 'seed': 42,
            'factors': list(factorial.CELLS['A']), 'command': command,
            'device': flags['--device'],
            'implementation_commit': 'fixture-implementation-commit',
            'source_commit': factorial.SOURCE_COMMIT,
            'data_artifacts': [
                {
                    'path': str(path), 'expected_sha256': expected,
                    'actual_sha256': expected, 'passed': True,
                }
                for path, expected in factorial.EXPECTED_DATA_ARTIFACTS['isolet'].items()
            ],
            'validation': {
                'dataset': 'isolet_vfl.npz-train',
                'logical_sha256': 'fixture-validation-hash',
                'split_seed': int(flags['--lambda_validation_split_seed']),
                'per_class': int(flags['--lambda_validation_per_class']),
                'evaluation_source': 'vector-train-validation',
            },
            'expected_tasks': 2 if smoke else 13,
            'selection_source': 'training-validation',
            'selection_test_used': False, 'test_used_for_fit': False,
            'smoke': smoke,
        }

    def _write_plan(self, command=None, smoke=False):
        path = self._root(smoke) / 'planned_protocol.json'
        self._write_json(path, self._plan_payload(command, smoke))
        return path

    def _write_launch_started(self, command=None, smoke=False):
        command = list(command or self._command(smoke))
        plan = self._root(smoke) / 'planned_protocol.json'
        return factorial._write_launch_started(
            self.spec, command, self._root(smoke), plan,
        )

    def _audit_cifar_smoke_with_manifest_label(self, label):
        original_spec = self.spec
        self.spec = 'cifar100:C:42'
        artifacts = {}
        for name in ('meta', 'test', 'train'):
            path = self.data / name
            path.write_bytes(name.encode('ascii'))
            artifacts[path] = self._hash(path)
        try:
            with mock.patch.object(
                factorial, 'EXPECTED_DATA_ARTIFACTS', {'cifar100': artifacts},
            ), mock.patch.object(
                factorial, 'EXPECTED_VALIDATION_HASHES',
                {'cifar100': 'fixture-validation-hash'},
            ):
                run = self._write_run(
                    smoke=True,
                    run=self._root(True) / 'factorial_cifar100_C_seed42_20260813_115506',
                )
                results_path = run / 'results.json'
                results = json.loads(results_path.read_text(encoding='utf-8'))
                results['selection_audit']['evaluation_source'] = (
                    'cifar100-train-validation'
                )
                self._write_json(results_path, results)
                self._write_json(run / 'validation/validation_manifest.json', {
                    'dataset': label,
                    'sha256': 'fixture-validation-hash',
                    'seed': 20260729,
                    'per_class': 25,
                })
                head_path = run / 'head_consolidation/event_1_CIL.json'
                head = json.loads(head_path.read_text(encoding='utf-8'))
                head.update({
                    'lr': 0.03, 'steps': 600,
                    'persistent_raw_example_count': 400,
                    'class_count': 20,
                })
                self._write_json(head_path, head)
                return self._audit(run, smoke=True)
        finally:
            self.spec = original_spec

    def test_audit_accepts_authoritative_cifar_manifest_label(self):
        record = self._audit_cifar_smoke_with_manifest_label('cifar100-train')
        self.assertTrue(record['checks']['validation'])

    def test_audit_does_not_mutate_matrix_tree(self):
        run = self._write_run()
        before = {
            path.relative_to(self.matrix_root): path.stat().st_mtime_ns
            for path in self.matrix_root.rglob('*')
        }
        matrix_mtime = self.matrix_root.stat().st_mtime_ns

        self._audit(run)

        self.assertEqual(self.matrix_root.stat().st_mtime_ns, matrix_mtime)
        self.assertEqual({
            path.relative_to(self.matrix_root): path.stat().st_mtime_ns
            for path in self.matrix_root.rglob('*')
        }, before)

    def test_audit_rejects_wrong_manifest_seed_or_per_class(self):
        for key, value in (('seed', 20260810), ('per_class', 41)):
            with self.subTest(key=key):
                run = self._write_run()
                manifest_path = run / 'validation/validation_manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest.update({'seed': 20260809, 'per_class': 40})
                manifest[key] = value
                self._write_json(manifest_path, manifest)
                with self.assertRaises(ValueError):
                    self._audit(run)

    def test_peer_audit_accepts_same_authoritative_manifest_label(self):
        original_spec = self.spec
        self.spec = 'cifar100:C:42'
        try:
            with mock.patch.object(
                factorial, 'EXPECTED_VALIDATION_HASHES',
                {'cifar100': 'fixture-validation-hash'},
            ):
                record = self._audit_cifar_smoke_with_manifest_label(
                    'cifar100-train'
                )
                root = self._root(True)
                factorial.atomic_json(root / 'record.json', record)
                (root / 'SUCCESS').touch()
                self.assertEqual(
                    factorial._peer_validation_hash(
                        root / 'record.json', self.matrix_root, 'cifar100',
                    ),
                    'fixture-validation-hash',
                )
        finally:
            self.spec = original_spec

    def test_peer_audit_rejects_wrong_manifest_seed_or_per_class(self):
        run = self._write_run()
        record = self._audit(run)
        root = self._root()
        factorial.atomic_json(root / 'record.json', record)
        (root / 'SUCCESS').touch()
        for key, value in (('seed', 1), ('per_class', 1)):
            with self.subTest(key=key):
                manifest_path = run / 'validation/validation_manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                manifest.update({'seed': 20260809, 'per_class': 40})
                manifest[key] = value
                self._write_json(manifest_path, manifest)
                config_path = run / 'config.json'
                config_payload = json.loads(config_path.read_text(encoding='utf-8'))
                config_payload[
                    'lambda_validation_split_seed'
                    if key == 'seed' else 'lambda_validation_per_class'
                ] = value
                self._write_json(config_path, config_payload)
                record['sha256']['validation_manifest'] = self._hash(manifest_path)
                record['sha256']['config'] = self._hash(config_path)
                factorial.atomic_json(root / 'record.json', record)
                self.assertIsNone(factorial._peer_validation_hash(
                    root / 'record.json', self.matrix_root, 'isolet',
                ))

    def test_audit_rejects_non_authoritative_cifar_manifest_labels(self):
        for label in (
            'cifar100', 'CIFAR100-train', 'cifar100-train ',
            'isolet_vfl.npz-train',
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._audit_cifar_smoke_with_manifest_label(label)

    def test_real_factorial_outputs_are_deterministically_discovered_and_audited(self):
        root = self._root()
        self._write_plan()
        self._write_launch_started()
        older = self._write_run(
            run=root / 'factorial_isolet_A_seed42_20260813_120000',
        )
        newer = self._write_run(
            run=root / 'factorial_isolet_A_seed42_20260813_120001',
        )
        self._write_run(run=root / 'unrelated_complete_directory')

        with mock.patch.object(
            factorial.subprocess, 'run', return_value=mock.Mock(returncode=0),
        ) as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0,
            )

        launch.assert_not_called()
        record = json.loads((root / 'record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['run_dir'], str(newer))
        self.assertNotEqual(record['run_dir'], str(older))
        self.assertEqual(
            factorial._peer_validation_hash(
                root / 'record.json', self.matrix_root, 'isolet',
            ),
            'fixture-validation-hash',
        )

    def test_real_factorial_output_symlink_is_rejected_before_launch(self):
        root = self._root()
        outside = self.root / 'outside-factorial-run'
        self._write_run(run=outside)
        root.mkdir(parents=True)
        (root / 'factorial_isolet_A_seed42_20260813_120000').symlink_to(
            outside, target_is_directory=True,
        )

        with mock.patch.object(
            factorial.subprocess, 'run', return_value=mock.Mock(returncode=0),
        ) as launch:
            with self.assertRaises(ValueError):
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root)

        launch.assert_not_called()

    def test_resume_rejects_other_factorial_cell_and_mismatched_legacy_run(self):
        root = self._root()
        matching = root / 'factorial_isolet_A_seed42_20260813_120000'
        wrong_factorial = root / 'factorial_isolet_B_seed42_20260813_120001'
        wrong_legacy = root / 'run_wrong_cell'
        for candidate in (matching, wrong_factorial, wrong_legacy):
            (candidate / 'checkpoints').mkdir(parents=True)
            (candidate / 'checkpoints' / 'resume_latest.pt').write_bytes(b'resume')
        self._write_json(matching / 'config.json', self._config(self._command()))
        wrong_command = factorial.build_command(
            'isolet:B:42', 'cuda:0', root,
        )
        wrong_config = self._config(wrong_command)
        self._write_json(wrong_factorial / 'config.json', wrong_config)
        self._write_json(wrong_legacy / 'config.json', wrong_config)

        def launch(command, **kwargs):
            self.assertEqual(
                as_flags(command)['--resume_run_dir'], str(matching),
            )
            self._write_run(command=command, run=matching)
            return mock.Mock(returncode=0)

        with mock.patch.object(factorial.subprocess, 'run', side_effect=launch):
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0,
            )

    def test_mismatched_complete_legacy_run_cannot_hide_legitimate_run(self):
        root = self._root()
        self._write_plan()
        self._write_launch_started()
        legitimate = self._write_run(run=root / 'run_a_legitimate')
        mismatched = self._write_run(run=root / 'run_z_wrong_cell')
        wrong_command = factorial.build_command(
            'isolet:B:42', 'cuda:0', root,
        )
        self._write_json(mismatched / 'config.json', self._config(wrong_command))

        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0,
            )

        launch.assert_not_called()
        record = json.loads((root / 'record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['run_dir'], str(legitimate))

    def test_complete_valid_job_is_audited_and_marked_success(self):
        self._write_plan()
        self._write_launch_started()
        run = self._write_run()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        launch.assert_not_called()
        root = self._root()
        record = json.loads((root / 'record.json').read_text(encoding='utf-8'))
        self.assertTrue(record['passed'])
        self.assertEqual(record['run_dir'], str(run))
        self.assertTrue((root / 'SUCCESS').is_file())
        self.assertFalse((root / 'FAILED.json').exists())

    def test_success_with_valid_record_is_skipped_without_subprocess(self):
        self._write_plan()
        self._write_launch_started()
        self._write_run()
        self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        launch.assert_not_called()

    def test_success_with_forged_record_fields_is_reaudited(self):
        self._write_plan()
        self._write_launch_started()
        self._write_run()
        self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        root = self._root()
        record_path = root / 'record.json'
        record = json.loads(record_path.read_text(encoding='utf-8'))
        record['implementation_commit'] = 'forged'
        record['validation_hash'] = 'forged'
        record['data_artifacts'][0]['actual_sha256'] = 'forged'
        self._write_json(record_path, record)
        with mock.patch.object(
            factorial, 'audit_completed_run', wraps=factorial.audit_completed_run,
        ) as audit, mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        self.assertEqual(audit.call_count, 2)
        launch.assert_not_called()
        rewritten = json.loads(record_path.read_text(encoding='utf-8'))
        self.assertEqual(rewritten['implementation_commit'], 'fixture-implementation-commit')
        self.assertEqual(rewritten['validation_hash'], 'fixture-validation-hash')
        self.assertTrue(all(item['passed'] for item in rewritten['data_artifacts']))

    def test_success_with_tampered_record_is_reaudited(self):
        self._write_plan()
        self._write_launch_started()
        self._write_run()
        self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        root = self._root()
        record_path = root / 'record.json'
        record = json.loads(record_path.read_text(encoding='utf-8'))
        outside = self.root / 'outside.pt'
        outside.write_bytes(b'outside')
        record['paths']['checkpoint'] = str(outside)
        record['sha256']['checkpoint'] = self._hash(outside)
        self._write_json(record_path, record)
        with mock.patch.object(
            factorial, 'audit_completed_run', wraps=factorial.audit_completed_run,
        ) as audit, mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        audit.assert_called_once()
        launch.assert_not_called()

    def test_incomplete_job_resumes_only_from_its_own_job_root(self):
        root = self._root()
        own = root / 'run_incomplete'
        alien = self.matrix_root / 'runs' / 'other_A_42' / 'run_newer'
        for candidate in (own, alien):
            (candidate / 'checkpoints').mkdir(parents=True)
            (candidate / 'checkpoints' / 'resume_latest.pt').write_bytes(b'resume')
        self._write_json(own / 'config.json', self._config(self._command()))
        self._write_json(alien / 'config.json', {'resume': True})

        def launch(command, **kwargs):
            self.assertEqual(as_flags(command)['--resume_run_dir'], str(own))
            self.assertEqual(kwargs['cwd'], factorial.WORKTREE)
            self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self._write_run(command=command, run=own)
            return mock.Mock(returncode=0)

        with mock.patch.object(factorial.subprocess, 'run', side_effect=launch):
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)

    def test_symlinked_roots_runs_and_resume_evidence_never_launch(self):
        cases = ('runs', 'job', 'run', 'config', 'resume')
        for case in cases:
            with self.subTest(case=case):
                matrix = self.root / f'matrix-{case}'
                outside = self.root / f'outside-{case}'
                outside.mkdir()
                job = matrix / 'runs' / 'isolet_A_42'
                if case == 'runs':
                    matrix.mkdir()
                    (matrix / 'runs').symlink_to(outside, target_is_directory=True)
                elif case == 'job':
                    (matrix / 'runs').mkdir(parents=True)
                    job.symlink_to(outside, target_is_directory=True)
                else:
                    job.mkdir(parents=True)
                    run = job / 'run_escape'
                    if case == 'run':
                        (outside / 'checkpoints').mkdir()
                        self._write_json(outside / 'config.json', {'resume': True})
                        (outside / 'checkpoints' / 'resume_latest.pt').write_bytes(b'resume')
                        run.symlink_to(outside, target_is_directory=True)
                    else:
                        (run / 'checkpoints').mkdir(parents=True)
                        if case == 'config':
                            external = outside / 'config.json'
                            self._write_json(external, {'resume': True})
                            (run / 'config.json').symlink_to(external)
                            (run / 'checkpoints' / 'resume_latest.pt').write_bytes(b'resume')
                        else:
                            self._write_json(run / 'config.json', {'resume': True})
                            external = outside / 'resume_latest.pt'
                            external.write_bytes(b'resume')
                            (run / 'checkpoints' / 'resume_latest.pt').symlink_to(external)
                with mock.patch.object(
                    factorial.subprocess, 'run',
                    return_value=mock.Mock(returncode=0),
                ) as launch:
                    with self.assertRaises(ValueError):
                        factorial.run_job(self.spec, 'cuda:0', matrix)
                launch.assert_not_called()
                self.assertFalse((outside / 'planned_protocol.json').exists())

    def test_symlinked_matrix_root_ancestor_never_writes_or_launches(self):
        actions = (
            ('run-job', lambda matrix: factorial.run_job(
                self.spec, 'cuda:0', matrix,
            )),
            ('reuse', factorial._reuse_write_targets),
            ('gate', factorial._gate_paths),
        )
        for name, action in actions:
            with self.subTest(name=name):
                outside = self.root / f'ancestor-target-{name}'
                outside.mkdir()
                linked = self.root / f'ancestor-link-{name}'
                linked.symlink_to(outside, target_is_directory=True)
                matrix = linked / 'matrix'
                with mock.patch.object(
                    factorial.subprocess, 'run',
                    return_value=mock.Mock(returncode=0),
                ) as launch:
                    with self.assertRaises(ValueError):
                        action(matrix)
                launch.assert_not_called()
                self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_write_targets_are_rejected_before_any_write_or_launch(self):
        for name in (
            'planned_protocol.json', 'job.log', 'FAILED.json',
            'record.json', 'SUCCESS',
        ):
            with self.subTest(name=name):
                matrix = self.root / ('matrix-write-' + name.replace('.', '-'))
                root = factorial.job_root(matrix, self.spec)
                root.mkdir(parents=True)
                outside = self.root / ('outside-write-' + name.replace('.', '-'))
                outside.write_text('unchanged', encoding='utf-8')
                (root / name).symlink_to(outside)
                with mock.patch.object(factorial.subprocess, 'run') as launch:
                    with self.assertRaises(ValueError):
                        factorial.run_job(self.spec, 'cuda:0', matrix)
                launch.assert_not_called()
                self.assertEqual(outside.read_text(encoding='utf-8'), 'unchanged')
                if name != 'planned_protocol.json':
                    self.assertFalse((root / 'planned_protocol.json').exists())

    def test_internal_symlinked_write_targets_are_rejected_before_any_write_or_launch(self):
        for name in (
            'planned_protocol.json', 'job.log', 'FAILED.json',
            'record.json', 'SUCCESS',
        ):
            with self.subTest(name=name):
                matrix = self.root / ('matrix-internal-' + name.replace('.', '-'))
                root = factorial.job_root(matrix, self.spec)
                root.mkdir(parents=True)
                internal = root / 'internal-target'
                internal.write_text('unchanged', encoding='utf-8')
                (root / name).symlink_to(internal.name)
                with mock.patch.object(factorial.subprocess, 'run') as launch:
                    with self.assertRaises(ValueError):
                        factorial.run_job(self.spec, 'cuda:0', matrix)
                launch.assert_not_called()
                self.assertEqual(internal.read_text(encoding='utf-8'), 'unchanged')

    def test_nonzero_subprocess_writes_failed_atomically_and_no_success(self):
        with mock.patch.object(
            factorial.subprocess, 'run', return_value=mock.Mock(returncode=17),
        ):
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 17)
        root = self._root()
        failed = json.loads((root / 'FAILED.json').read_text(encoding='utf-8'))
        self.assertEqual(failed['returncode'], 17)
        self.assertEqual(failed['spec'], self.spec)
        self.assertFalse((root / 'SUCCESS').exists())

    def test_zero_exit_without_complete_run_has_dedicated_failure_code(self):
        with mock.patch.object(
            factorial.subprocess, 'run', return_value=mock.Mock(returncode=0),
        ):
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 90)
        failed = json.loads((self._root() / 'FAILED.json').read_text(encoding='utf-8'))
        self.assertEqual(failed['returncode'], 90)
        self.assertFalse((self._root() / 'SUCCESS').exists())

    def test_missing_required_run_artifact_is_rejected(self):
        paths = (
            'checkpoints/event_12_CIL.pt', 'config.json', 'results.json',
            'validation/validation_manifest.json',
            'head_consolidation/event_12_CIL.json', 'data_flow_audit.jsonl',
        )
        for index, relative in enumerate(paths):
            with self.subTest(relative=relative):
                run = self._write_run()
                (run / relative).unlink()
                with self.assertRaises(ValueError):
                    self._audit(run)
                if index != len(paths) - 1:
                    self._write_run()

    def test_audit_rejects_outside_run_and_symlinked_evidence(self):
        run = self._write_run()
        outside_run = self.root / 'outside_run'
        run.rename(outside_run)
        with self.assertRaises(ValueError):
            self._audit(outside_run)

        run = self._write_run()
        checkpoint = run / 'checkpoints/event_12_CIL.pt'
        outside_checkpoint = self.root / 'outside.pt'
        outside_checkpoint.write_bytes(b'outside checkpoint')
        checkpoint.unlink()
        checkpoint.symlink_to(outside_checkpoint)
        with self.assertRaises(ValueError):
            self._audit(run)

    def test_audit_rejects_run_symlink_targeting_same_job_root(self):
        root = self._root()
        target = self._write_run(run=root / 'target_directory')
        linked = root / 'run_linked'
        linked.symlink_to(target, target_is_directory=True)

        with self.assertRaises(ValueError):
            self._audit(linked)

    def test_audit_rejects_command_not_built_for_its_spec(self):
        run = self._write_run()
        command = self._command()
        command[command.index('--batch_size') + 1] = '999'
        with self.assertRaises(ValueError):
            self._audit(run, command=command)

        command = self._command(resume=self.root / 'outside_resume')
        with self.assertRaises(ValueError):
            self._audit(run, command=command)

    def test_missing_or_nonnumeric_metric_is_rejected(self):
        for key, value in (
            ('AA_final', None), ('BWT', None), ('AA_final_taskil', None),
            ('AA_final', '0.7'),
        ):
            with self.subTest(key=key, value=value):
                run = self._write_run()
                results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
                if value is None:
                    results['cl_metrics'].pop(key)
                else:
                    results['cl_metrics'][key] = value
                self._write_json(run / 'results.json', results)
                with self.assertRaises(ValueError):
                    self._audit(run)

    def test_short_taskil_trajectory_or_wrong_final_task_is_rejected(self):
        run = self._write_run()
        results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
        results['cl_metrics']['AA_trajectory_taskil'].pop()
        self._write_json(run / 'results.json', results)
        with self.assertRaises(ValueError):
            self._audit(run)

        run = self._write_run()
        head_path = run / 'head_consolidation/event_12_CIL.json'
        head = json.loads(head_path.read_text(encoding='utf-8'))
        head['task_id'] = 11
        self._write_json(head_path, head)
        with self.assertRaises(ValueError):
            self._audit(run)

    def test_protocol_factor_data_validation_and_privacy_mismatches_reject(self):
        cases = (
            ('config.json', ('head_consolidation_mode',), 'task_class_bias'),
            ('head_consolidation/event_12_CIL.json', ('mode',), 'task_class_bias'),
            ('head_consolidation/event_12_CIL.json', ('lr',), 0.03),
            ('head_consolidation/event_12_CIL.json', ('steps',), 600),
            ('config.json', ('seed',), 43),
            ('config.json', ('data',), 'cifar100'),
            ('config.json', ('batch_size',), 999),
            ('config.json', ('replay_mode',), 'none'),
            ('head_consolidation/event_12_CIL.json', ('replay_selection',), 'random'),
            ('head_consolidation/event_12_CIL.json', ('samples_per_class',), 21),
            ('head_consolidation/event_12_CIL.json', ('persistent_embedding_count',), 1),
            ('head_consolidation/event_12_CIL.json', ('persistent_embedding_count',), False),
            ('head_consolidation/event_12_CIL.json', ('persistent_raw_example_count',), -1),
            ('head_consolidation/event_12_CIL.json', ('persistent_raw_example_count',), 521),
            ('head_consolidation/event_12_CIL.json', ('persistent_raw_example_count',), True),
            ('head_consolidation/event_12_CIL.json', ('class_count',), 0),
            ('head_consolidation/event_12_CIL.json', ('class_count',), 27),
            ('head_consolidation/event_12_CIL.json', ('class_count',), True),
            ('head_consolidation/event_12_CIL.json', ('schedule',), 'each_task'),
            ('results.json', ('selection_audit', 'passed'), False),
            ('results.json', ('selection_audit', 'test_used_for_selection'), True),
            ('results.json', ('selection_audit', 'evaluation_source'), 'test'),
            ('validation/validation_manifest.json', ('sha256',), 'wrong'),
            ('head_consolidation/event_12_CIL.json', ('test_used',), True),
            ('data_flow_audit.jsonl', ('test_used_for_fit',), True),
        )
        for relative, keys, value in cases:
            with self.subTest(relative=relative, keys=keys):
                self.npz.write_bytes(b'fixture npz')
                run = self._write_run()
                path = run / relative
                if path.suffix == '.jsonl':
                    payload = json.loads(path.read_text(encoding='utf-8'))
                else:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                target = payload
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                if path.suffix == '.jsonl':
                    path.write_text(json.dumps(payload) + '\n', encoding='utf-8')
                else:
                    self._write_json(path, payload)
                with self.assertRaises(ValueError):
                    self._audit(run)

        run = self._write_run()
        self.npz.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self._audit(run)

    def test_malformed_cached_record_types_never_escape_validation(self):
        self._write_plan()
        self._write_launch_started()
        run = self._write_run()
        self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        root = self._root()
        record_path = root / 'record.json'
        valid = json.loads(record_path.read_text(encoding='utf-8'))
        cases = (
            ('run_dir', {}), ('paths', []), ('sha256', []),
            ('checks', []), ('command', {}), ('implementation_commit', []),
        )
        for key, value in cases:
            with self.subTest(key=key):
                forged = dict(valid)
                forged[key] = value
                self._write_json(record_path, forged)
                with mock.patch.object(factorial.subprocess, 'run') as launch:
                    self.assertEqual(
                        factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0,
                    )
                launch.assert_not_called()
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8')), valid,
                )

    def test_forged_passed_peer_record_cannot_deny_legitimate_audit(self):
        peer = self.matrix_root / 'runs' / 'isolet_B_42'
        peer.mkdir(parents=True)
        self._write_json(peer / 'record.json', {
            'spec': 'isolet:B:42', 'dataset': 'isolet', 'seed': 42,
            'passed': True, 'validation_hash': 'forged-peer-hash',
        })
        (peer / 'SUCCESS').touch()
        run = self._write_run()
        record = self._audit(run)
        self.assertTrue(record['passed'])
        self.assertEqual(record['validation_hash'], 'fixture-validation-hash')

    def test_record_is_installed_before_success_and_contains_full_evidence(self):
        self._write_plan()
        self._write_launch_started()
        run = self._write_run()
        root = self._root()
        real_atomic = factorial.atomic_json

        def atomic(path, payload):
            if Path(path).name == 'record.json':
                self.assertFalse((root / 'SUCCESS').exists())
            return real_atomic(path, payload)

        with mock.patch.object(factorial, 'atomic_json', side_effect=atomic):
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)
        record = json.loads((root / 'record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['command'], self._command())
        self.assertEqual(record['implementation_commit'], 'fixture-implementation-commit')
        self.assertEqual(record['source_commit'], factorial.SOURCE_COMMIT)
        self.assertEqual(set(record['paths']), {
            'config', 'results', 'validation_manifest', 'head_audit',
            'checkpoint', 'data_flow_audit',
        })
        self.assertTrue(all(record['sha256'].values()))
        self.assertTrue(record['checks'])
        self.assertEqual(set(record['metrics']), {'AA_final', 'BWT', 'AA_final_taskil'})

    def test_planned_protocol_precedes_launch_and_contains_frozen_expectations(self):
        root = self._root()

        def launch(command, **kwargs):
            plan = json.loads((root / 'planned_protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(plan['command'], command)
            self.assertEqual(plan['device'], 'cuda:0')
            self.assertEqual(plan['implementation_commit'], 'fixture-implementation-commit')
            self.assertEqual(plan['source_commit'], factorial.SOURCE_COMMIT)
            self.assertEqual(plan['factors'], list(factorial.CELLS['A']))
            self.assertEqual(plan['expected_tasks'], 13)
            self.assertEqual(plan['selection_source'], 'training-validation')
            self.assertIs(plan['selection_test_used'], False)
            self.assertIs(plan['test_used_for_fit'], False)
            self.assertEqual(plan['validation']['logical_sha256'], 'fixture-validation-hash')
            self.assertEqual(plan['validation']['dataset'], 'isolet_vfl.npz-train')
            self.assertEqual(plan['validation']['split_seed'], 20260809)
            self.assertEqual(plan['validation']['per_class'], 40)
            self.assertTrue(all(item['passed'] for item in plan['data_artifacts']))
            self._write_run(command=command)
            return mock.Mock(returncode=0)

        with mock.patch.object(factorial.subprocess, 'run', side_effect=launch):
            self.assertEqual(factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0)

    def test_discovered_complete_run_without_plan_is_rejected_without_manufacturing_one(self):
        self._write_run()
        root = self._root()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()
        self.assertFalse((root / 'planned_protocol.json').exists())
        self.assertFalse((root / 'SUCCESS').exists())

    def test_retroactive_plan_and_launch_marker_cannot_adopt_completed_run(self):
        self._write_run()
        time.sleep(0.01)
        plan = self._write_plan()
        self._write_launch_started()
        plan_before = plan.read_bytes()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()
        self.assertEqual(plan.read_bytes(), plan_before)

    def test_completed_artifact_device_must_match_planned_device(self):
        self._write_plan()
        self._write_launch_started()
        run = self._write_run()
        config_path = run / 'config.json'
        payload = json.loads(config_path.read_text(encoding='utf-8'))
        payload['device'] = 'cuda:1'
        self._write_json(config_path, payload)
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()

    def test_forged_planned_command_is_rejected_without_overwrite_or_launch(self):
        command = self._command()
        command[command.index('--batch_size') + 1] = '999'
        plan_path = self._write_plan(command)
        self._write_launch_started(command)
        self._write_run()
        before = plan_path.read_bytes()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()
        self.assertEqual(plan_path.read_bytes(), before)

    def test_preplanned_device_must_match_claimed_device(self):
        plan_path = self._write_plan()
        self._write_launch_started()
        self._write_run()
        before = plan_path.read_bytes()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:1', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()
        self.assertEqual(plan_path.read_bytes(), before)

    def test_preplanned_implementation_mismatch_is_rejected(self):
        plan_path = self._write_plan()
        self._write_launch_started()
        self._write_run()
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        plan['implementation_commit'] = 'forged-implementation'
        self._write_json(plan_path, plan)
        before = plan_path.read_bytes()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root),
                factorial.AUDIT_MISMATCH,
            )
        launch.assert_not_called()
        self.assertEqual(plan_path.read_bytes(), before)

    def test_valid_preplanned_complete_run_is_adopted_without_plan_overwrite(self):
        plan_path = self._write_plan()
        self._write_launch_started()
        run = self._write_run()
        before = plan_path.read_bytes()
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root), 0,
            )
        launch.assert_not_called()
        self.assertEqual(plan_path.read_bytes(), before)
        record = json.loads((self._root() / 'record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['run_dir'], str(run))
        self.assertEqual(record['command'], self._command())

    def test_smoke_uses_own_root_two_tasks_and_one_epoch(self):
        root = self._root(smoke=True)

        def launch(command, **kwargs):
            flags = as_flags(command)
            self.assertEqual(flags['--num_tasks'], '2')
            self.assertEqual(flags['--epochs_per_task'], '1')
            self.assertEqual(flags['--results_dir'], str(root))
            self._write_run(smoke=True, command=command)
            return mock.Mock(returncode=0)

        with mock.patch.object(factorial.subprocess, 'run', side_effect=launch):
            self.assertEqual(
                factorial.run_job(self.spec, 'cuda:0', self.matrix_root, smoke=True), 0,
            )
        record = json.loads((root / 'record.json').read_text(encoding='utf-8'))
        self.assertTrue(record['smoke'])
        self.assertEqual(record['expected_tasks'], 2)
        self.assertFalse((self._root() / 'SUCCESS').exists())
        head_path = root / 'run_fixture/head_consolidation/event_1_CIL.json'
        head = json.loads(head_path.read_text(encoding='utf-8'))
        head['class_count'] = 26
        self._write_json(head_path, head)
        with self.assertRaises(ValueError):
            self._audit(root / 'run_fixture', smoke=True)

    def test_jobs_cli_partitions_exact_required_jobs_without_reuse_audit(self):
        self.matrix_root.mkdir(parents=True)
        required = ['isolet:A:42', 'cifar100:C:42', 'upmc_food101:D:42']
        self._write_json(self.matrix_root / 'required_jobs.json', {
            'required_jobs': required, 'reused_jobs': [],
        })
        stdout = io.StringIO()
        with mock.patch.object(factorial, 'run_job') as run, \
                mock.patch.object(factorial, 'audit_reuse') as reuse, \
                mock.patch.object(factorial.subprocess, 'run') as launch, \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(factorial.main([
                    'jobs', '--matrix-root', str(self.matrix_root),
                    '--worker', '1', '--workers', '2',
                ]), 0)
        reuse.assert_not_called()
        run.assert_not_called()
        launch.assert_not_called()
        self.assertEqual(stdout.getvalue().splitlines(), ['cifar100:C:42'])

    def test_check_and_jobs_paths_do_not_launch_as_an_import_side_effect(self):
        with mock.patch.object(factorial.subprocess, 'run') as launch:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(factorial.main(['check']), 0)
        launch.assert_not_called()
        self.assertIn('UNIFIED_HEAD_FACTORIAL_STATIC_CHECK_OK', stdout.getvalue())

    def test_cli_propagates_audit_failure_exit_code_without_launching(self):
        root = self._root()
        (root / 'run_incomplete' / 'checkpoints').mkdir(parents=True)
        self._write_json(root / 'run_incomplete' / 'results.json', {})
        (root / 'run_incomplete' / 'checkpoints' / 'event_12_CIL.pt').write_bytes(
            b'checkpoint'
        )
        completed = subprocess.run([
            str(factorial.PYTHON), str(Path(factorial.__file__)),
            'run-job', self.spec, '--device', 'cuda:0',
            '--matrix-root', str(self.matrix_root),
        ], cwd=factorial.WORKTREE, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, factorial.AUDIT_MISMATCH)
        self.assertTrue((root / 'FAILED.json').is_file())
        self.assertFalse((root / 'SUCCESS').exists())


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.matrix_root = self.root / 'matrix'
        self.commit = 'fixture-implementation-commit'
        self.validation_hashes = {
            dataset: f'{dataset}-validation-hash'
            for dataset in factorial.DATASET_NAMES
        }
        self.patchers = [
            mock.patch.object(
                factorial, '_implementation_commit', return_value=self.commit,
            ),
            mock.patch.object(
                factorial, 'EXPECTED_VALIDATION_HASHES', self.validation_hashes,
            ),
            mock.patch.object(factorial, '_valid_record', return_value=True),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    def _record(self, dataset, cell, metrics=None):
        return {
            'spec': f'{dataset}:{cell}:42',
            'dataset': dataset,
            'cell': cell,
            'seed': 42,
            'factors': list(factorial.CELLS[cell]),
            'implementation_commit': self.commit,
            'source_commit': factorial.SOURCE_COMMIT,
            'validation_hash': self.validation_hashes[dataset],
            'checks': {'strict_fixture': True},
            'metrics': metrics or {
                'AA_final': 0.70,
                'BWT': -0.10,
                'AA_final_taskil': 0.90,
            },
            'smoke': False,
            'passed': True,
        }

    def _write_run(self, dataset, cell, metrics=None):
        root = self.matrix_root / 'runs' / f'{dataset}_{cell}_42'
        record = self._record(dataset, cell, metrics)
        self._write_json(root / 'record.json', record)
        (root / 'SUCCESS').touch()
        return root / 'record.json'

    def _reuse_record(self, dataset, cell, passed=True):
        record = self._record(dataset, cell)
        record.pop('implementation_commit')
        record.pop('source_commit')
        record.pop('validation_hash')
        record.pop('factors')
        record.pop('smoke')
        record['expected_source_commit'] = factorial.SOURCE_COMMIT
        record['actual_source_commit'] = factorial.SOURCE_COMMIT
        record['validation_logical_hash'] = {
            'expected': self.validation_hashes[dataset],
            'actual': self.validation_hashes[dataset],
        }
        record['factor_values'] = {
            'expected': list(factorial.CELLS[cell]),
            'actual': list(factorial.CELLS[cell]),
        }
        record['passed'] = passed
        if not passed:
            record['checks'] = {'expected_rejection': False}
        return record

    def _complete(self, metrics=None):
        for dataset in factorial.DATASET_NAMES:
            for cell in factorial.CELLS:
                values = metrics(dataset, cell) if metrics else None
                self._write_run(dataset, cell, values)

    def _markers(self):
        return {
            name: (self.matrix_root / name).exists()
            for name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED')
        }

    def _summary(self):
        return json.loads((
            self.matrix_root / 'formal_report' / 'FACTORIAL_SUMMARY.json'
        ).read_text(encoding='utf-8'))

    def test_bad_or_incomplete_records_return_two_without_markers(self):
        cases = ('missing', 'duplicate', 'malformed', 'failed', 'stale',
                 'wrong-spec', 'invalid-evidence', 'path-escape')
        for case in cases:
            with self.subTest(case=case):
                matrix = self.root / f'matrix-{case}'
                self.matrix_root = matrix
                self._complete()
                target = matrix / 'runs' / 'cifar100_A_42'
                record_path = target / 'record.json'
                if case == 'missing':
                    record_path.unlink()
                elif case == 'duplicate':
                    duplicate = dict(self._record('cifar100', 'A'))
                    duplicate.pop('implementation_commit')
                    duplicate.pop('source_commit')
                    duplicate['expected_source_commit'] = factorial.SOURCE_COMMIT
                    duplicate['actual_source_commit'] = factorial.SOURCE_COMMIT
                    duplicate['validation_logical_hash'] = {
                        'expected': self.validation_hashes['cifar100'],
                        'actual': self.validation_hashes['cifar100'],
                    }
                    duplicate['factor_values'] = {
                        'expected': list(factorial.CELLS['A']),
                        'actual': list(factorial.CELLS['A']),
                    }
                    duplicate.pop('factors')
                    self._write_json(
                        matrix / 'reuse_audit' / 'cifar100_A.json', duplicate,
                    )
                    self.addCleanup(mock.patch.stopall)
                    mock.patch.object(
                        factorial, 'audit_reference', return_value=duplicate,
                    ).start()
                elif case == 'malformed':
                    record_path.write_text('{', encoding='utf-8')
                elif case == 'failed':
                    payload = json.loads(record_path.read_text(encoding='utf-8'))
                    payload['passed'] = False
                    self._write_json(record_path, payload)
                elif case == 'stale':
                    payload = json.loads(record_path.read_text(encoding='utf-8'))
                    payload['implementation_commit'] = 'stale'
                    self._write_json(record_path, payload)
                elif case == 'wrong-spec':
                    payload = json.loads(record_path.read_text(encoding='utf-8'))
                    payload['spec'] = 'cifar100:B:42'
                    self._write_json(record_path, payload)
                elif case == 'invalid-evidence':
                    factorial._valid_record.return_value = False
                else:
                    outside = self.root / f'outside-{case}'
                    target.rename(outside)
                    target.symlink_to(outside, target_is_directory=True)
                for marker in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED'):
                    (matrix / marker).touch()
                self.assertEqual(factorial.summarize(matrix), 2)
                self.assertEqual(self._markers(), {
                    'EXECUTION_SUCCESS': False,
                    'GATE_SUCCESS': False,
                    'GATE_FAILED': False,
                })
                if case == 'duplicate':
                    self.assertNotIn(
                        'cifar100:A:42',
                        {
                            item['spec']
                            for item in self._summary()['accepted_sources']
                        },
                    )
                factorial._valid_record.return_value = True

    def test_each_metric_independently_controls_eligibility(self):
        for metric in factorial.METRICS:
            with self.subTest(metric=metric):
                metrics = {
                    key: 0.60 for key in factorial.METRICS
                }
                incumbents = {
                    key: 0.70 for key in factorial.METRICS
                }
                by_key = {
                    (dataset, factorial.INCUMBENTS[dataset]): {
                        'metrics': dict(incumbents),
                    }
                    for dataset in factorial.DATASET_NAMES
                }
                for dataset in factorial.DATASET_NAMES:
                    candidate = dict(incumbents)
                    by_key[(dataset, 'C')] = {'metrics': candidate}
                by_key[('isolet', 'C')]['metrics'][metric] = 0.689
                result = factorial.eligibility('C', by_key)
                self.assertFalse(result['eligible'])
                self.assertEqual(result['failures'][0]['metric'], metric)

    def test_floor_is_inclusive_and_any_positive_epsilon_fails(self):
        by_key = {}
        for dataset in factorial.DATASET_NAMES:
            incumbent = factorial.INCUMBENTS[dataset]
            by_key[(dataset, incumbent)] = {
                'metrics': {key: 0.70 for key in factorial.METRICS},
            }
            by_key[(dataset, 'D')] = {
                'metrics': {key: 0.69 for key in factorial.METRICS},
            }
        self.assertTrue(factorial.eligibility('D', by_key)['eligible'])
        by_key[('upmc_food101', 'D')]['metrics']['BWT'] = 0.69 - 1e-12
        self.assertFalse(factorial.eligibility('D', by_key)['eligible'])

    def test_decimal_floor_is_not_rounded_down_before_comparison(self):
        incumbent_value = 0.28245932893118875
        exact_floor = Decimal('0.27245932893118875')
        by_key = {}
        for dataset in factorial.DATASET_NAMES:
            incumbent = factorial.INCUMBENTS[dataset]
            by_key[(dataset, incumbent)] = {'metrics': {
                key: incumbent_value for key in factorial.METRICS
            }}
            by_key[(dataset, 'D')] = {'metrics': {
                key: exact_floor for key in factorial.METRICS
            }}
        self.assertTrue(factorial.eligibility('D', by_key)['eligible'])
        rounded_lower = float(exact_floor)
        self.assertLess(Decimal(str(rounded_lower)), exact_floor)
        by_key[('isolet', 'D')]['metrics']['AA_final'] = rounded_lower
        self.assertFalse(factorial.eligibility('D', by_key)['eligible'])

    def test_negative_bwt_floor_is_exact(self):
        by_key = {}
        for dataset in factorial.DATASET_NAMES:
            incumbent = factorial.INCUMBENTS[dataset]
            by_key[(dataset, incumbent)] = {
                'metrics': {key: -0.09 for key in factorial.METRICS},
            }
            by_key[(dataset, 'D')] = {
                'metrics': {key: -0.10 for key in factorial.METRICS},
            }
        result = factorial.eligibility('D', by_key)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['datasets']['isolet']['floor']['BWT'], -0.10)

    def test_dataset_specific_hybrid_is_forbidden(self):
        def metrics(dataset, cell):
            values = {key: 0.70 for key in factorial.METRICS}
            if cell in ('C', 'D') and (
                (dataset == 'cifar100' and cell == 'D')
                or (dataset != 'cifar100' and cell == 'C')
            ):
                values['AA_final'] = 0.50
            return values

        self._complete(metrics)
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        summary = self._summary()
        self.assertNotIn('C', summary['eligible_cells'])
        self.assertNotIn('D', summary['eligible_cells'])

    def test_ranking_uses_macro_metrics_then_ascending_cell_name(self):
        orders = (
            ('AA_final', {'A': .73, 'B': .72, 'C': .71, 'D': .70}, 'A'),
            ('BWT', {'A': .20, 'B': .30, 'C': .10, 'D': .00}, 'B'),
            ('AA_final_taskil', {'A': .80, 'B': .80, 'C': .90, 'D': .70}, 'C'),
        )
        for index, (metric, values, expected) in enumerate(orders):
            matrix = self.root / f'rank-{index}'
            self.matrix_root = matrix
            def candidate(dataset, cell, chosen=metric, scores=values):
                payload = {key: .70 for key in factorial.METRICS}
                payload[chosen] = scores[cell]
                return payload
            self._complete(candidate)
            self.assertEqual(factorial.summarize(matrix), 0)
            self.assertEqual(self._summary()['selected_cell'], expected)

        self.matrix_root = self.root / 'rank-name'
        self._complete(lambda dataset, cell: {
            key: .70 for key in factorial.METRICS
        })
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        self.assertEqual(self._summary()['selected_cell'], 'A')

    def test_complete_no_eligible_writes_execution_and_gate_failed(self):
        def metrics(dataset, cell):
            value = .70 if cell == factorial.INCUMBENTS[dataset] else .50
            return {key: value for key in factorial.METRICS}
        self._complete(metrics)
        self.assertEqual(factorial.summarize(self.matrix_root), 3)
        self.assertEqual(self._markers(), {
            'EXECUTION_SUCCESS': True,
            'GATE_SUCCESS': False,
            'GATE_FAILED': True,
        })

    def test_eligible_complete_matrix_writes_success_and_removes_failed(self):
        self._complete()
        self.matrix_root.mkdir(parents=True, exist_ok=True)
        (self.matrix_root / 'GATE_FAILED').touch()
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        self.assertEqual(self._markers(), {
            'EXECUTION_SUCCESS': True,
            'GATE_SUCCESS': True,
            'GATE_FAILED': False,
        })

    def test_stale_marker_symlinks_are_rejected_without_touching_targets(self):
        for marker_name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED'):
            with self.subTest(marker=marker_name):
                self.matrix_root = self.root / f'matrix-{marker_name}'
                self._complete()
                external = self.root / f'outside-{marker_name}'
                external.write_bytes(b'unchanged')
                marker = self.matrix_root / marker_name
                marker.symlink_to(external)
                with self.assertRaises(ValueError):
                    factorial.summarize(self.matrix_root)
                self.assertTrue(marker.is_symlink())
                self.assertEqual(external.read_bytes(), b'unchanged')

    def test_test_only_metrics_cannot_alter_selection(self):
        def metrics(dataset, cell):
            return {
                'AA_final': .70 + .01 * list(factorial.CELLS).index(cell),
                'BWT': -.10,
                'AA_final_taskil': .90,
            }
        self._complete(metrics)
        for path in self.matrix_root.glob('runs/*/record.json'):
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['test_metrics'] = {
                'AA_final': 100 if payload['cell'] == 'A' else -100,
            }
            self._write_json(path, payload)
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        self.assertEqual(self._summary()['selected_cell'], 'D')
        self.assertFalse(self._summary()['test_used_for_selection'])

    def test_validation_hashes_are_frozen_and_identical_per_dataset(self):
        self._complete()
        path = self.matrix_root / 'runs' / 'isolet_D_42' / 'record.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['validation_hash'] = 'different'
        self._write_json(path, payload)
        self.assertEqual(factorial.summarize(self.matrix_root), 2)
        self.assertFalse(self._summary()['matrix_complete'])

    def test_reports_are_complete_deterministic_and_repeatable(self):
        self._complete()
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        report = self.matrix_root / 'formal_report'
        names = ('PER_CELL.csv', 'FACTORIAL_SUMMARY.json', 'GATE.json')
        before = {name: (report / name).read_bytes() for name in names}
        summary = json.loads(before['FACTORIAL_SUMMARY.json'])
        self.assertEqual(summary['seed'], 42)
        self.assertEqual(summary['tolerance'], .01)
        self.assertEqual(summary['selection_source'], 'training-validation')
        self.assertEqual(len(summary['accepted_sources']), 12)
        for cell in factorial.CELLS:
            item = summary['cells'][cell]
            self.assertEqual(item['factors']['mode'], factorial.CELLS[cell][0])
            self.assertIn('macro_metrics', item)
            self.assertIn('failures', item)
            for dataset in factorial.DATASET_NAMES:
                check = item['datasets'][dataset]
                self.assertEqual(set(check['candidate']), set(factorial.METRICS))
                self.assertEqual(set(check['floor']), set(factorial.METRICS))
                self.assertEqual(set(check['incumbent']), set(factorial.METRICS))
        self.assertEqual(factorial.summarize(self.matrix_root), 0)
        self.assertEqual(
            before, {name: (report / name).read_bytes() for name in names},
        )
        self.assertEqual(self._markers(), {
            'EXECUTION_SUCCESS': True,
            'GATE_SUCCESS': True,
            'GATE_FAILED': False,
        })

    def test_reuse_and_run_records_share_one_normalized_loader(self):
        self._complete()
        run_path = self.matrix_root / 'runs' / 'isolet_A_42' / 'record.json'
        reuse = self._reuse_record('isolet', 'A')
        reuse_path = self.matrix_root / 'reuse_audit' / 'isolet_A.json'
        self._write_json(reuse_path, reuse)
        run_path.unlink()
        (run_path.parent / 'SUCCESS').unlink()
        with mock.patch.object(
            factorial, 'audit_reference', return_value=reuse,
        ) as strict_reuse:
            loaded = factorial.load_matrix_records(self.matrix_root)
        self.assertTrue(loaded['complete'])
        self.assertEqual(len(loaded['by_key']), 12)
        self.assertEqual(
            loaded['by_key'][('isolet', 'A')]['source_type'], 'reuse',
        )
        self.assertEqual(
            loaded['by_key'][('isolet', 'B')]['source_type'], 'run',
        )
        strict_reuse.assert_called_once()

    def test_expected_failed_reuse_is_audit_evidence_not_a_loader_error(self):
        passed_reuse = {
            (dataset, cell): self._reuse_record(dataset, cell)
            for dataset in ('isolet', 'upmc_food101') for cell in ('A', 'B')
        }
        for (dataset, cell), record in passed_reuse.items():
            self._write_json(
                self.matrix_root / 'reuse_audit' / f'{dataset}_{cell}.json',
                record,
            )
        for cell in ('A', 'B'):
            self._write_json(
                self.matrix_root / 'reuse_audit' / f'cifar100_{cell}.json',
                self._reuse_record('cifar100', cell, passed=False),
            )
        for dataset in factorial.DATASET_NAMES:
            for cell in ('C', 'D'):
                self._write_run(dataset, cell)
        for cell in ('A', 'B'):
            self._write_run('cifar100', cell)

        with mock.patch.object(
            factorial, 'audit_reference',
            side_effect=lambda dataset, cell, root: passed_reuse[(dataset, cell)],
        ) as strict_reuse:
            self.assertEqual(factorial.summarize(self.matrix_root), 0)

        summary = self._summary()
        self.assertTrue(summary['matrix_complete'])
        self.assertEqual(len(summary['accepted_sources']), 12)
        self.assertEqual(
            {item['spec'] for item in summary['accepted_sources']},
            set(factorial.all_specs()),
        )
        self.assertEqual(
            [item['source_type'] for item in summary['accepted_sources']].count('reuse'),
            4,
        )
        self.assertEqual(
            [item['source_type'] for item in summary['accepted_sources']].count('run'),
            8,
        )
        self.assertTrue(all(
            (self.matrix_root / 'reuse_audit' / f'cifar100_{cell}.json').is_file()
            for cell in ('A', 'B')
        ))
        self.assertEqual(strict_reuse.call_count, 4)

    def test_reports_are_installed_before_root_markers(self):
        self._complete()
        original_touch = factorial._safe_touch
        observed = []
        report = self.matrix_root / 'formal_report'

        def observe_touch(path, *args, **kwargs):
            if path.name in ('EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED'):
                self.assertEqual(path.parent, self.matrix_root)
                self.assertTrue(all(
                    (report / name).is_file()
                    for name in ('PER_CELL.csv', 'FACTORIAL_SUMMARY.json', 'GATE.json')
                ))
                observed.append(path.name)
            return original_touch(path, *args, **kwargs)

        with mock.patch.object(factorial, '_safe_touch', new=observe_touch):
            self.assertEqual(factorial.summarize(self.matrix_root), 0)
        self.assertEqual(observed, ['EXECUTION_SUCCESS', 'GATE_SUCCESS'])

    def test_summarize_cli_propagates_gate_exit_code(self):
        self._complete(lambda dataset, cell: {
            key: (.70 if cell == factorial.INCUMBENTS[dataset] else .50)
            for key in factorial.METRICS
        })
        self.assertEqual(factorial.main([
            'summarize', '--matrix-root', str(self.matrix_root),
        ]), 3)


class LauncherContractTest(unittest.TestCase):
    MATRIX_NAME = 'unified_head_consolidation_factorial_seed42_20260813_120000'
    SCRIPT = Path(__file__).with_name(
        'run_unified_head_consolidation_factorial.sh',
    )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)

    def source_shell(self, body, env=None, timeout=None, cwd=None):
        command = (
            'set -euo pipefail; '
            'export LAUNCHER_SOURCE_ONLY=1; '
            f'source {shlex.quote(str(self.SCRIPT))}; {body}'
        )
        merged = os.environ.copy()
        merged['TEST_MATRIX_NAME'] = self.MATRIX_NAME
        if env:
            merged.update({key: str(value) for key, value in env.items()})
        return subprocess.run(
            ['bash', '-c', command], cwd=cwd or self.SCRIPT.parent,
            env=merged, text=True, capture_output=True, check=False,
            timeout=timeout,
        )

    def make_nvidia_smi(self):
        binary = self.temp / 'bin' / 'nvidia-smi'
        binary.parent.mkdir()
        binary.write_text(
            '#!/usr/bin/env bash\n'
            'case "$1" in\n'
            '  --query-gpu=*) printf "%b" "${MOCK_GPU_ROWS-}" ;;\n'
            '  --query-compute-apps=*) printf "%b" "${MOCK_APP_ROWS-}" ;;\n'
            '  *) exit 64 ;;\n'
            'esac\n',
            encoding='utf-8',
        )
        binary.chmod(0o755)
        return binary.parent

    def test_launcher_exists_is_executable_and_has_valid_bash_syntax(self):
        self.assertTrue(self.SCRIPT.is_file(), 'launcher script is absent')
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        completed = subprocess.run(
            ['bash', '-n', str(self.SCRIPT)], text=True,
            capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_constants_root_and_preflight_order_are_frozen(self):
        source = self.SCRIPT.read_text(encoding='utf-8')
        self.assertIn('set -euo pipefail', source)
        self.assertIn(
            'W=/home/chase/Yangxx/VF-CL/.worktrees/'
            'unified-head-consolidation-factorial', source,
        )
        self.assertIn('PY=/home/chase/anaconda3/envs/mlz_3.9/bin/python', source)
        self.assertIn(
            'SOURCE_COMMIT=a60d429c91966c5d42d057c8bd5a380777cc69cb',
            source,
        )
        self.assertIn('/home/chase/Yangxx/VF-CL/results/', source)
        ordered = [
            source.index('\n    preflight\n'),
            source.index('log "driver check"'),
            source.index('log "driver audit-reuse"'),
            source.index('\n    load_jobs\n'),
        ]
        self.assertEqual(ordered, sorted(ordered))

    def test_source_only_worktree_is_the_current_launcher_directory(self):
        completed = self.source_shell('printf %s "$W"')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, str(self.SCRIPT.parent.resolve()))

    def test_smoke_is_exactly_c_then_d_and_marker_follows_validation(self):
        source = self.SCRIPT.read_text(encoding='utf-8')
        start = source.index('run_smokes() {')
        end = source.index('\n}', start)
        smoke_calls = [line.strip() for line in source[start:end].splitlines()
                       if line.strip().startswith('run_smoke "cifar100:')]
        self.assertEqual(smoke_calls, [
            'run_smoke "cifar100:C:42" || return 1',
            'run_smoke "cifar100:D:42" || return 1',
        ])
        self.assertIn('--smoke', source)
        self.assertLess(
            source.index('completed_job "$spec" smoke'),
            source.index('touch -- "$ROOT/SMOKE_SUCCESS"'),
        )
        results = self.temp / 'results'
        results.mkdir()
        env = {'LAUNCHER_TEST_RESULTS_ROOT': results}
        failed = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root; '
            'run_smoke() { [[ "$1" != cifar100:D:42 ]]; }; '
            '! run_smokes; test ! -e "$ROOT/SMOKE_SUCCESS"', env,
        )
        self.assertEqual(failed.returncode, 0, failed.stderr)
        passed = self.source_shell(
            'ROOT="$RESULTS_BASE/${TEST_MATRIX_NAME%120000}120001"; prepare_root; '
            'run_smoke() { :; }; run_smokes; test -f "$ROOT/SMOKE_SUCCESS"',
            env,
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)

    def test_gpu_requires_numeric_threshold_and_no_active_vfcl_process(self):
        fake_bin = self.make_nvidia_smi()
        proc_root = self.temp / 'proc'
        (proc_root / '123').mkdir(parents=True)
        (proc_root / '123' / 'cmdline').write_bytes(
            b'/home/chase/Yangxx/VF-CL/main.py\x00--seed\x0042\x00',
        )
        base = {
            'PATH': f'{fake_bin}{os.pathsep}{os.environ["PATH"]}',
            'VFCL_PROC_ROOT': proc_root,
            'MOCK_GPU_ROWS': '0, GPU-a, 4000\\n1, GPU-b, 3499\\n',
            'MOCK_APP_ROWS': '',
        }
        available = self.source_shell('gpu_available 0', base)
        self.assertEqual(available.returncode, 0, available.stderr)
        too_low = self.source_shell('gpu_available 1', base)
        self.assertNotEqual(too_low.returncode, 0)
        active = dict(base, MOCK_APP_ROWS='GPU-a, 123\\n')
        blocked = self.source_shell('gpu_available 0', active)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn('VFCL', blocked.stderr)

        relative_proc = proc_root / '124'
        relative_proc.mkdir()
        (relative_proc / 'cmdline').write_bytes(b'python\x00main.py\x00')
        (relative_proc / 'cwd').symlink_to(
            '/home/chase/Yangxx/VF-CL/.worktrees/'
            'unified-head-consolidation-factorial',
        )
        relative = self.source_shell('gpu_available 0', dict(
            base, MOCK_APP_ROWS='GPU-a, 124\\n',
        ))
        self.assertNotEqual(relative.returncode, 0)
        self.assertIn('VFCL', relative.stderr)

    def test_malformed_or_missing_gpu_evidence_is_unavailable(self):
        fake_bin = self.make_nvidia_smi()
        base = {
            'PATH': f'{fake_bin}{os.pathsep}{os.environ["PATH"]}',
            'VFCL_PROC_ROOT': self.temp / 'proc',
            'MOCK_APP_ROWS': '',
        }
        malformed = self.source_shell(
            'gpu_available 0', dict(base, MOCK_GPU_ROWS='not,csv\\n'),
        )
        self.assertNotEqual(malformed.returncode, 0)
        internal_space = self.source_shell(
            'gpu_available 0', dict(base, MOCK_GPU_ROWS='0, GPU-a, 3 500\\n'),
        )
        self.assertNotEqual(internal_space.returncode, 0)
        missing_proc = self.source_shell('gpu_available 0', dict(
            base,
            MOCK_GPU_ROWS='0, GPU-a, 4000\\n',
            MOCK_APP_ROWS='GPU-a, 999999\\n',
        ))
        self.assertNotEqual(missing_proc.returncode, 0)

    def test_root_helpers_reject_outside_and_symlinked_targets(self):
        results = self.temp / 'results'
        results.mkdir()
        env = {'LAUNCHER_TEST_RESULTS_ROOT': results}
        safe = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root; '
            'test -d "$ROOT/logs"; test -d "$ROOT/claims"', env,
        )
        self.assertEqual(safe.returncode, 0, safe.stderr)
        outside = self.source_shell(
            'ROOT="${RESULTS_BASE}_outside"; prepare_root', env,
        )
        self.assertNotEqual(outside.returncode, 0)

        target = results / 'target'
        target.mkdir()
        link = results / 'unified_head_consolidation_factorial_seed42_20260813_120002'
        link.symlink_to(target, target_is_directory=True)
        linked = self.source_shell(
            'ROOT="$RESULTS_BASE/unified_head_consolidation_factorial_seed42_20260813_120002"; prepare_root', env,
        )
        self.assertNotEqual(linked.returncode, 0)

        matrix = results / 'unified_head_consolidation_factorial_seed42_20260813_120003'
        matrix.mkdir()
        (matrix / 'SMOKE_SUCCESS').symlink_to(self.temp / 'elsewhere')
        marker = self.source_shell(
            'ROOT="$RESULTS_BASE/unified_head_consolidation_factorial_seed42_20260813_120003"; prepare_root', env,
        )
        self.assertNotEqual(marker.returncode, 0)

        for index, child in enumerate(('logs', 'claims'), start=4):
            name = f'unified_head_consolidation_factorial_seed42_20260813_12000{index}'
            matrix = results / name
            matrix.mkdir()
            (matrix / child).symlink_to(target, target_is_directory=True)
            linked_child = self.source_shell(
                f'ROOT="$RESULTS_BASE/{name}"; prepare_root', env,
            )
            self.assertNotEqual(linked_child.returncode, 0)

    def test_root_basename_ownership_and_restart_identity_are_exact(self):
        results = self.temp / 'results'
        results.mkdir()
        env = {'LAUNCHER_TEST_RESULTS_ROOT': results}
        wrong = self.source_shell(
            'ROOT="$RESULTS_BASE/matrix"; prepare_root', env,
        )
        self.assertNotEqual(wrong.returncode, 0)

        unowned_name = 'unified_head_consolidation_factorial_seed42_20260813_120010'
        (results / unowned_name).mkdir()
        unowned = self.source_shell(
            f'ROOT="$RESULTS_BASE/{unowned_name}"; prepare_root', env,
        )
        self.assertNotEqual(unowned.returncode, 0)

        created = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root; '
            'test -f "$ROOT/CODE_COMMIT.txt"; '
            'test -f "$ROOT/MATRIX_IDENTITY.txt"; '
            'prepare_root', env,
        )
        self.assertEqual(created.returncode, 0, created.stderr)

        root = results / self.MATRIX_NAME
        identity = root / 'MATRIX_IDENTITY.txt'
        original = identity.read_bytes()
        identity.write_text('forged\n', encoding='utf-8')
        forged = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root', env,
        )
        self.assertNotEqual(forged.returncode, 0)
        identity.write_bytes(original)
        (root / 'CODE_COMMIT.txt').write_text('0' * 40 + '\n', encoding='utf-8')
        wrong_commit = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root', env,
        )
        self.assertNotEqual(wrong_commit.returncode, 0)

    def test_every_launcher_runtime_target_rejects_symlinks(self):
        results = self.temp / 'results'
        results.mkdir()
        target_dir = self.temp / 'target-dir'
        target_dir.mkdir()
        target_file = self.temp / 'target-file'
        target_file.write_text('unchanged', encoding='utf-8')
        directories = ('logs', 'claims', 'reuse_audit', 'runs', 'smoke', 'formal_report')
        files = (
            'required_jobs.json', 'SMOKE_SUCCESS', 'FAILED_JOB', 'STOPPED',
            'worker_0.pid', 'worker_1.pid', 'WORKERS_READY',
            'EXECUTION_SUCCESS', 'GATE_SUCCESS', 'GATE_FAILED',
        )
        for index, name in enumerate((*directories, *files), start=20):
            with self.subTest(name=name):
                root_name = f'unified_head_consolidation_factorial_seed42_20260813_12{index:04d}'
                root = results / root_name
                root.mkdir()
                (root / 'CODE_COMMIT.txt').write_text(
                    subprocess.check_output(
                        ['git', '-C', str(self.SCRIPT.parent), 'rev-parse', 'HEAD'],
                        text=True,
                    ),
                    encoding='utf-8',
                )
                (root / 'MATRIX_IDENTITY.txt').write_text(
                    'unified_head_consolidation_factorial:seed42:v1\n',
                    encoding='utf-8',
                )
                (root / name).symlink_to(
                    target_dir if name in directories else target_file,
                    target_is_directory=name in directories,
                )
                completed = self.source_shell(
                    f'ROOT="$RESULTS_BASE/{root_name}"; prepare_root',
                    {'LAUNCHER_TEST_RESULTS_ROOT': results},
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(target_file.read_text(encoding='utf-8'), 'unchanged')

    def test_claims_are_atomic_and_completed_jobs_are_not_reclaimed(self):
        results = self.temp / 'results'
        results.mkdir()
        root = results / self.MATRIX_NAME
        mock_python = self.temp / 'mock-python'
        mock_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
        mock_python.chmod(0o755)
        env = {'LAUNCHER_TEST_RESULTS_ROOT': results}
        body = (
            f'PY={shlex.quote(str(mock_python))}; '
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root; '
            'claim_job "cifar100:C:42"; '
            '! claim_job "cifar100:C:42"; '
            'release_claim "cifar100:C:42"; '
            'job="$ROOT/runs/cifar100_C_42"; mkdir -p "$job"; '
            'printf \'{"passed":true,"spec":"cifar100:C:42",'
            '"smoke":false}\\n\' > "$job/record.json"; '
            'touch "$job/SUCCESS"; '
            'completed_job "cifar100:C:42" full; '
            '! claim_job "cifar100:C:42"'
        )
        completed = self.source_shell(body, env)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse((root / 'claims' / 'cifar100_C_42.claim').exists())

        external = self.temp / 'external-claim'
        external.mkdir()
        owner = external / 'owner'
        owner.write_text('do not remove', encoding='utf-8')
        claim = root / 'claims' / 'isolet_C_42.claim'
        claim.symlink_to(external, target_is_directory=True)
        linked = self.source_shell(
            f'PY={shlex.quote(str(mock_python))}; '
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; ! claim_job "isolet:C:42"', env,
        )
        self.assertEqual(linked.returncode, 0, linked.stderr)
        self.assertEqual(owner.read_text(encoding='utf-8'), 'do not remove')

    def test_claim_outcomes_distinguish_acquired_complete_live_and_unsafe(self):
        results = self.temp / 'results'
        results.mkdir()
        body = r'''
ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"
prepare_root
completed_job() { [[ "${MOCK_COMPLETE:-0}" == 1 ]]; }
set +e
claim_job 'cifar100:C:42'; acquired=$?
claim_job 'cifar100:C:42'; live=$?
release_claim 'cifar100:C:42'
MOCK_COMPLETE=1
claim_job 'cifar100:C:42'; complete=$?
MOCK_COMPLETE=0
mkdir -p "$ROOT/claims/external"
ln -s "$ROOT/claims/external" "$ROOT/claims/isolet_C_42.claim"
claim_job 'isolet:C:42'; unsafe=$?
set -e
printf 'claim-statuses acquired=%s complete=%s live=%s unsafe=%s\n' \
    "$acquired" "$complete" "$live" "$unsafe" >&2
test "$acquired" -eq 0
test "$complete" -eq 10
test "$live" -eq 11
test "$unsafe" -eq 12
'''
        completed = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_in_progress_claim_without_owner_is_never_stolen(self):
        results = self.temp / 'results'
        results.mkdir()
        cases = (
            (
                'job',
                "claim=$ROOT/claims/cifar100_C_42.claim; mkdir -- \"$claim\"; "
                "set +e; claim_job 'cifar100:C:42'; status=$?; set -e; "
                'test "$status" -eq "$CLAIM_UNSAFE"; '
                'test -d "$claim"; test ! -e "$claim/owner"',
            ),
            (
                'launcher',
                'claim=$ROOT/claims/launcher.claim; mkdir -- "$claim"; '
                'set +e; claim_launcher; status=$?; set -e; '
                'test "$status" -ne 0; '
                'test -d "$claim"; test ! -e "$claim/owner"',
            ),
        )
        for index, (name, assertion) in enumerate(cases, start=1):
            with self.subTest(name=name):
                root_name = f'{self.MATRIX_NAME[:-1]}{index}'
                completed = self.source_shell(
                    f'ROOT="$RESULTS_BASE/{root_name}"; prepare_root; {assertion}',
                    {'LAUNCHER_TEST_RESULTS_ROOT': results},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_worker_never_skips_live_or_unsafe_claims(self):
        results = self.temp / 'results'
        results.mkdir()
        for status in (11, 12):
            with self.subTest(status=status):
                body = rf'''
ROOT="$RESULTS_BASE/${{TEST_MATRIX_NAME%120000}}1200{status}"
prepare_root
touch "$ROOT/WORKERS_READY"
completed_job() {{ return 1; }}
claim_job() {{ return {status}; }}
wait_gpu() {{ touch "$ROOT/unexpected-launch"; }}
record_failure() {{ :; }}
terminate_sibling() {{ :; }}
set +e
worker_loop 0 'cifar100:C:42'
worker_status=$?
set -e
test "$worker_status" -eq {status}
test ! -e "$ROOT/unexpected-launch"
'''
                completed = self.source_shell(
                    body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_final_required_jobs_are_strictly_revalidated(self):
        accepted = self.source_shell(r'''
REQUIRED_JOBS=('cifar100:C:42' 'isolet:C:42')
completed_job() { return 0; }
validate_required_jobs
''')
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        rejected = self.source_shell(r'''
REQUIRED_JOBS=('cifar100:C:42' 'isolet:C:42')
completed_job() { [[ "$1" == 'cifar100:C:42' ]]; }
validate_required_jobs
''')
        self.assertNotEqual(rejected.returncode, 0)

    def test_driver_partitions_failure_traps_and_gate_semantics_are_explicit(self):
        source = self.SCRIPT.read_text(encoding='utf-8')
        self.assertIn('driver jobs --matrix-root "$ROOT"', source)
        self.assertIn('--worker 0 --workers 2', source)
        self.assertIn('--worker 1 --workers 2', source)
        self.assertIn('required_jobs.json', source)
        self.assertIn('all_specs', source)
        self.assertNotRegex(source, r'JOBS=\([^)]*cifar100')
        self.assertIn('FAILED_JOB', source)
        self.assertIn('STOPPED', source)
        self.assertIn('terminate_sibling', source)
        self.assertIn("trap 'terminate_children; release_launcher' EXIT", source)
        self.assertIn('tracked_driver summarize --matrix-root "$ROOT"', source)
        self.assertIn('driver._valid_record', source)
        self.assertRegex(source, r'3\)\s+return 3')
        self.assertLess(
            source.index('wait "$WORKER_0_PID"'),
            source.index('\n    summarize_gate\n'),
        )

    def test_job_loader_accepts_only_complete_disjoint_driver_partitions(self):
        results = self.temp / 'results'
        results.mkdir()
        root = results / self.MATRIX_NAME
        root.mkdir()
        required = [
            'cifar100:C:42', 'cifar100:D:42',
            'isolet:C:42', 'isolet:D:42',
        ]
        (root / 'required_jobs.json').write_text(
            json.dumps({'required_jobs': required}), encoding='utf-8',
        )
        mock_driver = r'''
driver() {
    if [[ " $* " == *" --worker 0 "* ]]; then
        printf '%s\n' cifar100:C:42 isolet:C:42
    elif [[ " $* " == *" --worker 1 "* ]]; then
        printf '%s\n' cifar100:D:42 isolet:D:42
    else
        printf '%s\n' cifar100:C:42 cifar100:D:42 isolet:C:42 isolet:D:42
    fi
}
'''
        body = (
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; ' + mock_driver +
            'load_jobs; test "${#REQUIRED_JOBS[@]}" -eq 4; '
            'test "${#WORKER_0_JOBS[@]}" -eq 2; '
            'test "${#WORKER_1_JOBS[@]}" -eq 2'
        )
        accepted = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

        duplicate_driver = mock_driver.replace(
            'cifar100:C:42 cifar100:D:42 isolet:C:42 isolet:D:42',
            'cifar100:C:42 cifar100:C:42 isolet:C:42 isolet:D:42',
        )
        rejected = self.source_shell(
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; ' + duplicate_driver + 'load_jobs',
            {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertNotEqual(rejected.returncode, 0)

    def test_first_failure_records_exact_evidence_and_terminates_sibling(self):
        results = self.temp / 'results'
        results.mkdir()
        body = r'''
ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"
prepare_root
rm -f "$ROOT/WORKERS_READY"
LAUNCHER_SOURCE_ONLY=1 setsid bash -c 'source "$1"; ROOT=$2; worker_loop 1 "${@:3}"' bash \
    "$W/run_unified_head_consolidation_factorial.sh" "$ROOT" & sibling=$!
trap 'terminate_group "$sibling"' EXIT
atomic_text "$ROOT/worker_1.pid" "$(worker_identity "$sibling" 1)"
record_failure 'isolet:D:42' 0 17
terminate_sibling 0
set +e
wait "$sibling"
set -e
test -f "$ROOT/STOPPED"
test "$(<"$ROOT/FAILED_JOB")" = '{"spec":"isolet:D:42","worker":0,"exit":17}'
trap - EXIT
'''
        completed = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_source_only_main_locks_worker_cwd_to_worktree(self):
        results = self.temp / 'results'
        results.mkdir()
        caller = self.temp / 'caller'
        caller.mkdir()
        body = r'''
ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"
prepare_root() { mkdir -p "$ROOT/logs"; }
claim_launcher() { :; }
release_launcher() { :; }
prepare_workers() { :; }
preflight() { :; }
tracked_driver() { :; }
run_smokes() { :; }
load_jobs() { WORKER_0_JOBS=(); WORKER_1_JOBS=(); }
validate_required_jobs() { :; }
summarize_gate() { :; }
main
test "$(pwd)" = "$W"
test -s "$ROOT/worker_0.pid"
test -s "$ROOT/worker_1.pid"
'''
        completed = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results}, timeout=3, cwd=caller,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_unrelated_process_group_cannot_be_claimed_as_worker(self):
        results = self.temp / 'results'
        results.mkdir()
        body = r'''
ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"
prepare_root
setsid bash -c 'exec tail -f /dev/null' & unrelated=$!
trap 'terminate_group "$unrelated"' EXIT
set +e
worker_identity "$unrelated" 1 > "$ROOT/worker_1.pid"
identity_status=$?
set -e
test "$identity_status" -ne 0
terminate_sibling 0
kill -0 "$unrelated"
terminate_group "$unrelated"
trap - EXIT
'''
        completed = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_sibling_termination_revalidates_pid_start_pgid_and_role(self):
        results = self.temp / 'results'
        results.mkdir()
        for index, corruption in enumerate(('start', 'pgid', 'role'), start=1):
            with self.subTest(corruption=corruption):
                body = rf'''
ROOT="$RESULTS_BASE/${{TEST_MATRIX_NAME%120000}}12010{index}"
prepare_root
rm -f "$ROOT/WORKERS_READY"
LAUNCHER_SOURCE_ONLY=1 setsid bash -c 'source "$1"; ROOT=$2; worker_loop 1 "${{@:3}}"' bash \
    "$W/run_unified_head_consolidation_factorial.sh" "$ROOT" & sibling=$!
trap 'terminate_group "$sibling"' EXIT
identity=$(worker_identity "$sibling" 1)
case {shlex.quote(corruption)} in
  start) identity=$(printf '%s\n' "$identity" | awk -F: '{{print $1 ":1:" $3 ":" $4 ":" $5}}') ;;
  pgid) identity=$(printf '%s\n' "$identity" | awk -F: '{{print $1 ":" $2 ":1:" $4 ":" $5}}') ;;
  role) identity=$(printf '%s\n' "$identity" | awk -F: '{{print $1 ":" $2 ":" $3 ":worker_0:" $5}}') ;;
esac
atomic_text "$ROOT/worker_1.pid" "$identity"
terminate_sibling 0
kill -0 "$sibling"
terminate_group "$sibling"
trap - EXIT
'''
                completed = self.source_shell(
                    body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_termination_kills_and_waits_for_the_entire_process_group(self):
        child_file = self.temp / 'child.pid'
        body = (
            'setsid bash -c '
            + shlex.quote(f'sleep 30 & echo $! > {child_file}; wait')
            + ' & group=$!; '
            f'while [[ ! -s {shlex.quote(str(child_file))} ]]; do :; done; '
            'terminate_group "$group"; '
            '! kill -0 -- "-$group" 2>/dev/null'
        )
        completed = self.source_shell(body)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cleanup_waits_only_for_registered_groups(self):
        fifo = self.temp / 'launcher-log.fifo'
        log = self.temp / 'launcher.log'
        body = rf'''
mkfifo {shlex.quote(str(fifo))}
{{ printf ready; sleep 2; }} > {shlex.quote(str(fifo))} & writer=$!
tee -a {shlex.quote(str(log))} < {shlex.quote(str(fifo))} >/dev/null & logger=$!
setsid bash -c 'sleep 30 & wait' & ACTIVE_MAIN_PID=$!
setsid bash -c 'sleep 30 & wait' & WORKER_0_PID=$!
start=$(date +%s%N)
terminate_children
elapsed_ms=$((($(date +%s%N) - start) / 1000000))
! kill -0 -- "-$ACTIVE_MAIN_PID" 2>/dev/null
! kill -0 -- "-$WORKER_0_PID" 2>/dev/null
test "$elapsed_ms" -lt 1000
kill -0 "$logger"
kill -0 "$writer"
wait "$writer"
wait "$logger"
'''
        completed = self.source_shell(body)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exit_cleanup_preserves_nonzero_status_without_waiting_on_tee(self):
        results = self.temp / 'results'
        results.mkdir()
        log = self.temp / 'exit.log'
        body = (
            'ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"; prepare_root; claim_launcher; '
            f'exec > >(tee -a {shlex.quote(str(log))}); '
            "trap 'terminate_children; release_launcher' EXIT; exit 17"
        )
        try:
            completed = self.source_shell(
                body, {'LAUNCHER_TEST_RESULTS_ROOT': results}, timeout=2,
            )
        except subprocess.TimeoutExpired:
            self.fail('EXIT cleanup waited on the logging process substitution')
        self.assertEqual(completed.returncode, 17, completed.stderr)
        self.assertFalse(
            (results / self.MATRIX_NAME / 'claims' / 'launcher.claim').exists(),
        )

    def test_restart_coordination_clears_stale_files_but_rejects_live_owner(self):
        results = self.temp / 'results'
        results.mkdir()
        body = r'''
ROOT="$RESULTS_BASE/$TEST_MATRIX_NAME"
prepare_root
printf '999999\n' > "$ROOT/worker_0.pid"
printf '999998\n' > "$ROOT/worker_1.pid"
touch "$ROOT/WORKERS_READY"
claim_launcher
prepare_workers
test ! -e "$ROOT/worker_0.pid"
test ! -e "$ROOT/worker_1.pid"
test ! -e "$ROOT/WORKERS_READY"
release_launcher
mkdir -p "$ROOT/claims/launcher.claim"
process_identity "$BASHPID" > "$ROOT/claims/launcher.claim/owner"
! claim_launcher
'''
        completed = self.source_shell(
            body, {'LAUNCHER_TEST_RESULTS_ROOT': results},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_tracked_driver_returns_gate_exit_three_without_exiting_shell(self):
        mock_python = self.temp / 'exit-three'
        mock_python.write_text('#!/usr/bin/env bash\nexit 3\n', encoding='utf-8')
        mock_python.chmod(0o755)
        completed = self.source_shell(
            f'PY={shlex.quote(str(mock_python))}; '
            'set +e; tracked_driver summarize; '
            'status=$?; set -e; test "$status" -eq 3; printf survived',
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, 'survived')

        gate = self.source_shell(
            'ROOT=/unused; tracked_driver() { return 3; }; '
            'set +e; summarize_gate; status=$?; set -e; test "$status" -eq 3',
        )
        self.assertEqual(gate.returncode, 0, gate.stderr)

    def test_forbidden_launch_paths_and_destructive_operations_are_absent(self):
        source = self.SCRIPT.read_text(encoding='utf-8').lower()
        for forbidden in (
            'seed43', 'seed44', 'password', 'private key', 'download',
            'begin openssh private key', 'id_rsa', 'preprocess', 'rm -rf',
            'run_cifar100_external',
        ):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
