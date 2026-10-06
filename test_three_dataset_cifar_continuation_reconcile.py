import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

import three_dataset_seed42_reconcile as legacy
import three_dataset_formal_driver as driver
import three_dataset_cifar_continuation_reconcile as continuation
from three_dataset_cifar_continuation_reconcile import (
    _allowed_change, build_bundle, inspect_origin, origin_keys,
    load_bundle, publish_bundle, require_origin_membership, validate_bundle,
)


class CifarOriginMembershipTests(unittest.TestCase):
    def test_only_reviewed_operational_extensions_join_the_migration_scope(self):
        for name in ('three_dataset_cifar_continuation_report.py',
                     'run_three_dataset_formal_comparison.sh',
                     'prune_completed_runs.py',
                     'three_dataset_resource_gate.py',
                     'vfl_trainer.py', 'cl_methods/adagauss.py'):
            with self.subTest(name=name):
                self.assertTrue(_allowed_change(name))
        for name in ('runner.py', 'models.py', 'config.py',
                     'three_dataset_formal_metrics.py'):
            with self.subTest(name=name):
                self.assertFalse(_allowed_change(name))

    def test_exact_eighteen_and_no_gpm(self):
        keys = origin_keys()
        self.assertEqual(len(keys), 18)
        self.assertEqual(len(set(keys)), 18)
        self.assertNotIn('cifar100:gpm:42', keys)
        require_origin_membership({quote(key, safe='') + '.json' for key in keys})

    def test_missing_extra_or_gpm_record_rejected(self):
        names = {quote(key, safe='') + '.json' for key in origin_keys()}
        for changed in (names - {next(iter(names))},
                        names | {'cifar100%3Agpm%3A42.json'},
                        names | {'unrelated.json'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                require_origin_membership(changed)


@unittest.skipUnless(os.environ.get('VFCL_CIFAR18_ORIGIN'),
                     'real v8 origin path is required')
class RealCifarOriginTests(unittest.TestCase):
    def test_exact_eighteen_completed_records_are_verified(self):
        rows, source = inspect_origin(Path(os.environ['VFCL_CIFAR18_ORIGIN']))
        self.assertEqual([row['spec_key'] for row in rows], list(origin_keys()))
        self.assertEqual({row['origin_source_commit'] for row in rows},
                         {'a7915143129d986f4561b93b49119fd4cfcd79f6'})
        self.assertEqual(source['plan_sha256'],
                         'fd6a746fc9b1c5f91bb78bd844563cfe64e80b2b2225b2b721ba6b4d82efd110')


@unittest.skipUnless(os.environ.get('VFCL_CIFAR18_ORIGIN'),
                     'real v8 origin path is required')
class CifarBundleTests(unittest.TestCase):
    def test_bundle_has_exact_origin_and_current_identity(self):
        bundle = build_bundle(
            Path(os.environ['VFCL_CIFAR18_ORIGIN']),
            Path(__file__).resolve().parent,
        )
        self.assertEqual(len(bundle['admitted']), 18)
        self.assertEqual([row['spec_key'] for row in bundle['admitted']],
                         list(origin_keys()))
        self.assertEqual(bundle['rejected'], [])
        self.assertEqual(bundle['ambiguous'], [])
        self.assertEqual(bundle['sources'][0]['source_commit'],
                         'a7915143129d986f4561b93b49119fd4cfcd79f6')
        with self.assertRaises(ValueError):
            validate_bundle({**bundle, 'admitted':
                             bundle['admitted'] + [bundle['admitted'][0]]})
        forged = copy.deepcopy(bundle)
        forged['admitted'][0]['spec_key'] = 'cifar100:gpm:42'
        with self.assertRaises(ValueError):
            validate_bundle(forged)

    def test_unreviewed_scientific_path_is_rejected(self):
        self.assertFalse(_allowed_change('models.py'))
        self.assertFalse(_allowed_change('runner.py'))


class CrossCommitMetricTests(unittest.TestCase):
    def test_old_and_new_frozen_state_metrics_match(self):
        probe = r'''
import contextlib, io, json, random, tempfile
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from test_adaptive_deferred_evaluation import DeferredEvaluationTests, _FormalStateTrainer
from adaptive_consolidation_audit import evaluate_formal_deferred_trajectory
random.seed(42); np.random.seed(42); torch.manual_seed(42)
with tempfile.TemporaryDirectory() as tmp:
    with contextlib.redirect_stdout(io.StringIO()):
        case = DeferredEvaluationTests()
        args, paths, final_path, tasks, cache = case._formal_evaluation_fixture(Path(tmp))
        payloads = {
            'snapshots': [torch.load(path, map_location='cpu', weights_only=True)
                          for path in paths],
            'final': torch.load(final_path, map_location='cpu', weights_only=True),
        }
        with mock.patch('adaptive_consolidation_audit._fresh_trainer',
                        side_effect=lambda _payload, _args: _FormalStateTrainer()):
            result = evaluate_formal_deferred_trajectory(
                args=args, snapshot_paths=paths, final_checkpoint=final_path,
                task_classes=tasks, cached_test_batches=cache, output_dir=tmp,
                recompute_payloads=payloads)
    print(json.dumps({name: result[name] for name in
                      ('cl_metrics', 'task_acc_history')},
                     sort_keys=True, separators=(',', ':')))
'''
        env = {**os.environ,
               'VFCL_EXPERIMENT_PROFILE': 'single-dataset-full-matrix',
               'VFCL_FORMAL_DATASET': 'cifar100',
               'VFCL_PYTHON': '/home/c3080/YangXiaoXiang/envs/vfcl/bin/python'}
        old = Path('/home/c3080/YangXiaoXiang/VF-CL-worktrees/formal-pipelined-audit-3080')
        new = Path(__file__).resolve().parent
        outputs = []
        for source in (old, new):
            completed = subprocess.run(
                [env['VFCL_PYTHON'], '-c', probe], cwd=source, env=env,
                capture_output=True, text=True, check=True)
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])


@unittest.skipUnless(os.environ.get('VFCL_CIFAR18_ORIGIN'),
                     'real v8 origin path is required')
class CifarBundlePublicationTests(unittest.TestCase):
    def test_publication_is_immutable_and_tamper_evident(self):
        worktree = Path(__file__).resolve().parent
        bundle = build_bundle(Path(os.environ['VFCL_CIFAR18_ORIGIN']),
                              worktree)
        real_git = legacy.git

        def clean_for_tdd(path, *args):
            if args == ('status', '--porcelain'):
                return ''
            return real_git(path, *args)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                legacy, 'git', side_effect=clean_for_tdd):
            output = Path(tmp) / 'reuse'
            publish_bundle(bundle, output)
            self.assertEqual(load_bundle(output), bundle)
            self.assertEqual({path.name for path in output.iterdir()}, {
                'CIFAR_CONTINUATION_REUSE.json',
                'CIFAR_CONTINUATION_REUSE_AUDIT.json',
                'CIFAR_CONTINUATION_REUSE_SUCCESS',
            })
            self.assertTrue(all((path.stat().st_mode & 0o777) == 0o444
                                for path in output.iterdir()))
            target = output / 'CIFAR_CONTINUATION_REUSE.json'
            altered = json.loads(target.read_text(encoding='utf-8'))
            altered['admitted'][0]['metrics']['aa_final'] += 0.01
            target.chmod(0o644)
            target.write_bytes(legacy.canonical(altered) + b'\n')
            target.chmod(0o444)
            with self.assertRaises(ValueError):
                load_bundle(output)


class CifarGitScopeTests(unittest.TestCase):
    def test_bundle_cannot_understate_actual_changed_paths(self):
        proof_path = Path(
            '/home/c3080/YangXiaoXiang/VF-CL/results/'
            'cifar18-origin-proof-20260930-v1'
        )
        with self.assertRaisesRegex(ValueError, 'commit differs'):
            load_bundle(proof_path)
        with legacy.Evidence(proof_path) as proof:
            bundle = proof.json('CIFAR_CONTINUATION_REUSE.json')
            proof.verify()
        worktree = Path(__file__).resolve().parent
        head = legacy.git(worktree, 'rev-parse', 'HEAD').strip()
        actual = sorted(set(legacy.git(
            worktree, 'diff', '--name-only',
            'a7915143129d986f4561b93b49119fd4cfcd79f6', head,
        ).splitlines()))
        self.assertGreater(len(actual), 1)
        bundle['current_commit'] = head
        bundle['changed_paths'] = actual[1:]
        migration = legacy.digest({
            'origin_commit': 'a7915143129d986f4561b93b49119fd4cfcd79f6',
            'current_commit': head,
            'changed_paths': bundle['changed_paths'],
            'protocol_sha256': bundle['sources'][0]['protocol_sha256'],
            'data_sha256': bundle['sources'][0]['data_sha256'],
        })
        bundle['migration_sha256'] = migration
        for row in bundle['admitted']:
            row['migration_sha256'] = migration
        validate_bundle(bundle)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'reuse'
            output.mkdir(mode=0o700)
            audit = continuation._audit_payload(bundle)
            driver.install_json_exclusive(output / 'CIFAR_CONTINUATION_REUSE.json',
                                          bundle)
            driver.install_json_exclusive(
                output / 'CIFAR_CONTINUATION_REUSE_AUDIT.json', audit)
            driver.install_json_exclusive(
                output / 'CIFAR_CONTINUATION_REUSE_SUCCESS',
                continuation._success_payload(bundle, audit))
            with self.assertRaisesRegex(ValueError, 'Git scope differs'):
                load_bundle(output)
