import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replay11_matrix as matrix
from test_replay11_evidence import (
    _archive_manifest, _canonical_sha, _job, _sha, _write,
)
from adaptive_consolidation_audit import FORMAL_SOURCE_FILES
from replay11_evidence import (
    CORRECTED_COMMIT, PRODUCER_COMMIT, equivalence_certificate,
    merge_attestations, verify_archive_manifest, verify_local_job,
)


def _corrected_job(root):
    job = _job(root)
    run = job / 'run'
    plan_path = job / 'JOB_PLAN.json'
    plan = json.loads(plan_path.read_text())
    plan.update(code_commit=CORRECTED_COMMIT,
                pilot_deferred=False, pilot_bic=False)
    plan['command'] = matrix_command = __import__(
        'replay11_evidence')._committed_matrix(CORRECTED_COMMIT).command_for(
            ('cifar100', 'mir', 42), job,
            python=Path('python'), worktree=Path('/tmp/vfcl-replay11-code99'),
            device='cuda:0', data_root=root / 'data')
    _write(plan_path, plan)
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    for index in range(2, len(matrix_command), 2):
        name = matrix_command[index][2:]
        token = matrix_command[index + 1]
        kind = matrix.registry.OPTION_SCHEMA.get(name, 'str')
        config[name] = ([int(token)] if name == 'unlearn_after_tasks' else
                        [[int(token)]] if name == 'unlearn_classes' else
                        int(token) if kind == 'int' else
                        float(token) if kind == 'float' else
                        token == '1' if kind == 'bool' else token)
    _write(config_path, config)
    source_hashes = {
        name: hashlib.sha256(subprocess.check_output([
            'git', '-C', str(Path(__file__).parent), 'show',
            f'{CORRECTED_COMMIT}:{name}'])).hexdigest()
        for name in FORMAL_SOURCE_FILES
    }
    provenance = {'schema_version': 1, 'source_commit': CORRECTED_COMMIT,
                  'source_sha256': source_hashes}
    result_path = run / 'results.json'
    result = json.loads(result_path.read_text())
    result['config'] = dict(config)
    result['source_provenance'] = provenance
    _write(result_path, result)
    frozen_path = run / 'FORMAL_STATE_FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    identity = frozen['identity']
    identity['source_commit'] = CORRECTED_COMMIT
    identity['source_provenance'] = provenance
    identity['protocol']['source_provenance'] = provenance
    identity['protocol_sha256'] = _canonical_sha(identity['protocol'])
    _write(frozen_path, frozen)
    freeze = {'path': 'FORMAL_STATE_FROZEN.json', 'sha256': _sha(frozen_path)}
    transaction = _canonical_sha({'identity': identity, 'freeze': freeze})
    for name in ('PENDING', 'CONSUMING', 'COMPLETE'):
        path = run / f'FORMAL_EVALUATION_{name}.json'
        value = json.loads(path.read_text())
        value.update(identity=identity, freeze=freeze,
                     transaction_sha256=transaction)
        _write(path, value)
    sealed_path = run / 'FORMAL_EVALUATION_SEALED.json'
    sealed = json.loads(sealed_path.read_text())
    sealed.update(transaction_sha256=transaction, freeze=freeze,
                  source_provenance=provenance,
                  identity_sha256=_canonical_sha(identity),
                  protocol_sha256=identity['protocol_sha256'])
    for name in ('PENDING', 'CONSUMING', 'COMPLETE'):
        sealed[name.lower()]['sha256'] = _sha(
            run / f'FORMAL_EVALUATION_{name}.json')
    _write(sealed_path, sealed)
    for name in ('PUBLISHING', 'PUBLISHED'):
        path = run / f'FORMAL_EVALUATION_{name}.json'
        value = json.loads(path.read_text())
        value.update(transaction_sha256=transaction,
                     source_provenance=provenance,
                     results_sha256=_canonical_sha(result))
        value['seal']['sha256'] = _sha(sealed_path)
        if name == 'PUBLISHED':
            value['results']['sha256'] = _sha(result_path)
        _write(path, value)
    success_path = job / 'JOB_SUCCESS.json'
    success = json.loads(success_path.read_text())
    success.update(plan)
    success['results_sha256'] = _sha(result_path)
    _write(success_path, success)
    return job


class Replay11V3Test(unittest.TestCase):
    def test_corrected_full_transaction_attests_real_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            record = verify_local_job(_corrected_job(Path(directory)))
            self.assertEqual(record['code_commit'], CORRECTED_COMMIT)
            self.assertEqual(record['admission'], 'formal')
            self.assertEqual(record['source_sha256']['cl_methods/mir.py'],
                             __import__('replay11_evidence')._producer_source_sha256(
                                 CORRECTED_COMMIT)['cl_methods/mir.py'])
            self.assertEqual(len(record['transaction_sha256']), 64)
            self.assertEqual(record['data_hashes'],
                             matrix.expected_data('cifar100'))
            self.assertEqual(len(record['command_sha256']), 64)

    def test_corrected_producer_requires_matching_full_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            for name in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = job / name
                value = json.loads(path.read_text())
                value['code_commit'] = CORRECTED_COMMIT
                value['pilot_deferred'] = False
                value['pilot_bic'] = False
                _write(path, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_archive_cannot_relabel_old_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            manifest = _archive_manifest(job)
            record = json.loads(manifest.read_text())
            record['code_commit'] = CORRECTED_COMMIT
            _write(manifest, record)
            with self.assertRaises(ValueError):
                verify_archive_manifest(job, manifest)

    def test_legacy_reuse_certificate_for_five_methods(self):
        for method in ('balanced_contrast', 'er', 'der_pp', 'er_ace', 'adagauss'):
            for dataset in matrix.DATASETS:
                job = (dataset, method, 42)
                command = matrix.command_for(
                    job, Path('/results'), python=Path('/python'),
                    worktree=Path('/source'), device='cuda:0',
                    data_root=Path('/data'))
                certificate = equivalence_certificate(job, command)
                self.assertTrue(certificate['equivalent'])
                self.assertEqual(certificate['method'], method)
                self.assertEqual(certificate['normalized_command']['cl_method'],
                                 method)
                self.assertEqual(set(certificate['command_generator_sha256']),
                                 {PRODUCER_COMMIT, CORRECTED_COMMIT})
                self.assertIn(f'cl_methods/{method}.py',
                              certificate['source_sha256'])

    def test_old_changed_method_cannot_claim_equivalence(self):
        job = ('cifar100', 'mir', 42)
        command = matrix.command_for(
            job, Path('/results'), python=Path('/python'),
            worktree=Path('/source'), device='cuda:0',
            data_root=Path('/data'))
        with self.assertRaises(ValueError):
            equivalence_certificate(job, command)

    def test_changed_observed_command_cannot_claim_equivalence(self):
        job = ('cifar100', 'er', 42)
        command = matrix.command_for(
            job, Path('/results'), python=Path('/python'),
            worktree=Path('/source'), device='cuda:0',
            data_root=Path('/data'))
        command[command.index('--epochs_per_task') + 1] = '1'
        with self.assertRaises(ValueError):
            equivalence_certificate(job, command)

    def test_changed_vector_cache_cannot_claim_equivalence(self):
        job = ('isolet', 'er', 42)
        command = matrix.command_for(
            job, Path('/results'), python=Path('/python'),
            worktree=Path('/source'), device='cuda:0',
            data_root=Path('/data'))
        command[command.index('--vector_npz') + 1] = '/other/file.npz'
        with self.assertRaises(ValueError):
            equivalence_certificate(job, command)

    def test_changed_shared_blob_cannot_claim_equivalence(self):
        import replay11_evidence as evidence
        job = ('cifar100', 'er', 42)
        command = matrix.command_for(
            job, Path('/results'), python=Path('/python'),
            worktree=Path('/source'), device='cuda:0',
            data_root=Path('/data'))
        actual_hashes = evidence._producer_source_sha256

        def changed(commit):
            hashes = dict(actual_hashes(commit))
            if commit == CORRECTED_COMMIT:
                hashes['runner.py'] = '0' * 64
            return hashes

        with patch.object(evidence, '_producer_source_sha256', changed):
            with self.assertRaisesRegex(ValueError, 'source blobs'):
                equivalence_certificate(job, command)

    def test_both_pilot_flags_excluded_from_formal_job(self):
        for flag in ('pilot_deferred', 'pilot_bic'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                job = _job(Path(directory))
                for name in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                    path = job / name
                    value = json.loads(path.read_text())
                    value[flag] = True
                    _write(path, value)
                with self.assertRaisesRegex(ValueError, 'pilot'):
                    verify_local_job(job)

    def test_legacy_changed_method_is_historical_only(self):
        with tempfile.TemporaryDirectory() as directory:
            record = verify_local_job(_job(Path(directory)))
            self.assertEqual(record['code_commit'], PRODUCER_COMMIT)
            self.assertEqual(record['admission'], 'historical')

    def test_catalog_needs_certificate_for_old_reusable_rows(self):
        rows = [
            {'job': f'{dataset}/{method}/{seed}',
             'code_commit': CORRECTED_COMMIT, 'admission': 'formal',
             'metrics': {'aa_final_cil': .5, 'bwt_cil': -.1,
                         'task_il_final': .7},
             'results_sha256': 'a' * 64,
             'checkpoint_sha256': 'b' * 64}
            for dataset, method, seed in matrix.jobs()
        ]
        old = next(row for row in rows if row['job'] == 'cifar100/er/42')
        old['code_commit'] = PRODUCER_COMMIT
        with self.assertRaises(ValueError):
            merge_attestations(rows)
        command = matrix.command_for(
            ('cifar100', 'er', 42), Path('/results'),
            python=Path('/python'), worktree=Path('/source'),
            device='cuda:0', data_root=Path('/data'))
        old['reuse_certificate'] = equivalence_certificate(
            ('cifar100', 'er', 42), command)
        self.assertEqual(merge_attestations(rows)['jobs'], 99)
        old['pilot_deferred'] = True
        with self.assertRaises(ValueError):
            merge_attestations(rows)


if __name__ == '__main__':
    unittest.main()
