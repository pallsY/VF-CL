import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import replay11_matrix as matrix
from replay11_evidence import (
    InconclusiveEvidence, verify_archive_manifest, verify_local_job,
)


COMMIT = '99e81b28e0544e182f777a34d7e7a786a1c3b920'
MAIN_BLOB_SHA = '115e8c0ac31054de1ff657b5b4f5024d088017c4b3bb4d0b2019ec84d8ea4e89'


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value):
    content = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
        separators=(',', ': ')) + '\n'
    return hashlib.sha256(content.encode()).hexdigest()


def _job(root, dataset='cifar100'):
    tasks = {'cifar100': 10, 'isolet': 13}[dataset]
    job = root / dataset / 'mir' / '42'
    run = job / 'run'
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'frozen-final-checkpoint')
    history = [
        {
            'step': f'event_{i}_CIL',
            'per_task_accs': (
                {f'task_{j}': 0.5 - 0.01 * j for j in range(tasks)}
                if i == tasks - 1 else {f'task_{i}': 0.8}
            ),
            **({'per_task_accs_taskil': {
                f'task_{j}': 0.7 for j in range(tasks)
            }} if i == tasks - 1 else {}),
        }
        for i in range(tasks)
    ]
    frozen_snapshots = []
    for i in range(tasks):
        snapshot = run / 'formal_snapshots' / f'event_{i}_CIL.pt'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(f'snapshot-{i}'.encode())
        source = run / 'checkpoints' / f'event_{i}_CIL.pt'
        source.write_bytes(f'source-{i}'.encode())
        frozen_snapshots.append({
            'file': {
                'path': f'formal_snapshots/event_{i}_CIL.pt',
                'sha256': _sha(snapshot),
            },
            'source_file': {
                'path': f'checkpoints/event_{i}_CIL.pt',
                'sha256': _sha(source),
            },
        })
    config = {
        'cl_method': 'mir', 'seed': 42, 'num_tasks': tasks,
        'num_parties': 4, 'formal_deferred_evaluation': True,
    }
    provenance = {
        'source_commit': COMMIT,
        'source_sha256': {
            'main.py': MAIN_BLOB_SHA,
        },
    }
    result = {
        'config': dict(config), 'task_acc_history': history,
        'source_provenance': provenance,
    }
    _write(run / 'config.json', config)
    _write(run / 'results.json', result)
    frozen_identity = {
        'source_commit': COMMIT, 'snapshots': frozen_snapshots,
    }
    _write(run / 'FORMAL_STATE_FROZEN.json', {
        'schema_version': 1, 'status': 'frozen',
        'identity': frozen_identity,
    })
    freeze = {
        'path': 'FORMAL_STATE_FROZEN.json',
        'sha256': _sha(run / 'FORMAL_STATE_FROZEN.json'),
    }
    transaction = _canonical_sha({
        'identity': frozen_identity, 'freeze': freeze})
    evaluation_payload = {'history': history}
    evaluation = _canonical_sha(evaluation_payload)
    _write(run / 'FORMAL_EVALUATION_PENDING.json', {
        'schema_version': 1, 'status': 'pending',
        'identity': frozen_identity, 'freeze': freeze,
        'transaction_sha256': transaction,
    })
    _write(run / 'FORMAL_EVALUATION_CONSUMING.json', {
        'schema_version': 1, 'status': 'consuming',
        'identity': frozen_identity, 'freeze': freeze,
        'transaction_sha256': transaction,
    })
    _write(run / 'FORMAL_EVALUATION_COMPLETE.json', {
        'schema_version': 1, 'status': 'complete',
        'identity': frozen_identity, 'freeze': freeze,
        'transaction_sha256': transaction,
        'evaluation_sha256': evaluation,
        'evaluation': evaluation_payload,
    })
    record = lambda name: {
        'path': name + '.json', 'sha256': _sha(run / (name + '.json')),
    }
    _write(run / 'FORMAL_EVALUATION_SEALED.json', {
        'schema_version': 1, 'status': 'sealed',
        'transaction_sha256': transaction, 'freeze': freeze,
        'pending': record('FORMAL_EVALUATION_PENDING'),
        'consuming': record('FORMAL_EVALUATION_CONSUMING'),
        'complete': record('FORMAL_EVALUATION_COMPLETE'),
        'source_provenance': provenance,
    })
    seal = record('FORMAL_EVALUATION_SEALED')
    _write(run / 'FORMAL_EVALUATION_PUBLISHING.json', {
        'schema_version': 1, 'status': 'publishing',
        'transaction_sha256': transaction,
        'evaluation_sha256': evaluation,
        'results_sha256': _canonical_sha(result), 'seal': seal,
        'source_provenance': provenance,
    })
    _write(run / 'FORMAL_EVALUATION_PUBLISHED.json', {
        'schema_version': 1, 'status': 'published',
        'transaction_sha256': transaction,
        'evaluation_sha256': evaluation,
        'results_sha256': _canonical_sha(result), 'seal': seal,
        'source_provenance': provenance,
        'checkpoint': {'path': 'checkpoints/formal_final.pt',
                       'sha256': _sha(checkpoint)},
        'results': {'path': 'results.json',
                    'sha256': _sha(run / 'results.json')},
    })
    _write(run / 'formal_access' / 'test.consumed.json', {
        'schema_version': 1, 'status': 'consumed',
        'access': {'split': 'test', 'event_idx': tasks - 1,
                   'phase': 'final_test_post_install'},
    })
    command = matrix.command_for(
        (dataset, 'mir', 42), job,
        python=Path('python'), worktree=Path('/tmp/vfcl-replay11-code99'),
        device='cuda:0', data_root=root / 'data')
    for i in range(2, len(command), 2):
        key, token = command[i][2:], command[i + 1]
        kind = matrix.registry.OPTION_SCHEMA.get(key, 'str')
        if key == 'unlearn_after_tasks':
            config[key] = [int(token)]
        elif key == 'unlearn_classes':
            config[key] = [[int(token)]]
        elif kind == 'bool':
            config[key] = token == '1'
        elif kind == 'int':
            config[key] = int(token)
        elif kind == 'float':
            config[key] = float(token)
        else:
            config[key] = token
    _write(run / 'config.json', config)
    plan = {
        'dataset': dataset, 'method': 'mir', 'seed': 42,
        'smoke': False, 'code_commit': COMMIT, 'command': command,
        'data_hashes': matrix.expected_data(dataset),
    }
    _write(job / 'JOB_PLAN.json', plan)
    _write(job / 'JOB_SUCCESS.json', {
        **plan,
        'metrics': matrix.metrics_from_history(
            history, expected_tasks=tasks, deferred=True),
        'results_sha256': _sha(run / 'results.json'),
        'checkpoint_sha256': _sha(checkpoint),
    })
    return job


def _archive_manifest(job):
    rows = []
    for path in sorted(job.rglob('*')):
        if path.is_file():
            rows.append([
                path.relative_to(job).as_posix(),
                path.stat().st_size, _sha(path),
            ])
    digest = hashlib.sha256(json.dumps(
        rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    manifest = job.parent / 'ARCHIVE_MANIFEST.json'
    _write(manifest, {
        'job': 'cifar100/mir/42', 'code_commit': COMMIT,
        'source': '/source/cifar100/mir/42',
        'destination': str(job),
        'files': len(rows), 'bytes': sum(row[1] for row in rows),
        'source_tree_sha256': digest,
        'destination_tree_sha256': digest,
    })
    return manifest


class Replay11EvidenceTest(unittest.TestCase):
    def test_valid_job_has_portable_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            attestation = verify_local_job(_job(Path(directory)))
            self.assertEqual(attestation['job'], 'cifar100/mir/42')
            self.assertEqual(attestation['code_commit'], COMMIT)
            self.assertEqual(set(attestation['metrics']), {
                'aa_final_cil', 'bwt_cil', 'task_il_final'})

    def test_result_summary_may_omit_party_count(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            results = job / 'run' / 'results.json'
            value = json.loads(results.read_text())
            value['config'].pop('num_parties')
            value['config'].pop('formal_deferred_evaluation')
            _write(results, value)
            success = job / 'JOB_SUCCESS.json'
            record = json.loads(success.read_text())
            record['results_sha256'] = _sha(results)
            _write(success, record)
            published = job / 'run' / 'FORMAL_EVALUATION_PUBLISHED.json'
            marker = json.loads(published.read_text())
            marker['results']['sha256'] = _sha(results)
            marker['results_sha256'] = _canonical_sha(value)
            _write(published, marker)
            publishing = job / 'run' / 'FORMAL_EVALUATION_PUBLISHING.json'
            marker = json.loads(publishing.read_text())
            marker['results_sha256'] = _canonical_sha(value)
            _write(publishing, marker)
            self.assertEqual(
                verify_local_job(job)['job'], 'cifar100/mir/42')

    def test_changed_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            result = job / 'run' / 'results.json'
            result.write_bytes(result.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_changed_frozen_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            snapshot = job / 'run' / 'formal_snapshots' / 'event_4_CIL.pt'
            snapshot.write_bytes(b'changed-after-freeze')
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_changed_source_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            checkpoint = job / 'run' / 'checkpoints' / 'event_4_CIL.pt'
            checkpoint.write_bytes(b'changed-after-freeze')
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_unpublished_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            marker = job / 'run' / 'FORMAL_EVALUATION_PUBLISHED.json'
            value = json.loads(marker.read_text())
            value['status'] = 'pending'
            _write(marker, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_missing_complete_transaction_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            (job / 'run' / 'FORMAL_EVALUATION_COMPLETE.json').unlink()
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_broken_transaction_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            marker = job / 'run' / 'FORMAL_EVALUATION_PUBLISHED.json'
            value = json.loads(marker.read_text())
            value['transaction_sha256'] = 'd' * 64
            _write(marker, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_forged_consistent_transaction_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            run = job / 'run'
            for name in ('PENDING', 'CONSUMING', 'COMPLETE'):
                path = job / 'run' / f'FORMAL_EVALUATION_{name}.json'
                value = json.loads(path.read_text())
                value['transaction_sha256'] = 'd' * 64
                _write(path, value)
            sealed = run / 'FORMAL_EVALUATION_SEALED.json'
            value = json.loads(sealed.read_text())
            value['transaction_sha256'] = 'd' * 64
            for name in ('pending', 'consuming', 'complete'):
                value[name]['sha256'] = _sha(
                    run / f'FORMAL_EVALUATION_{name.upper()}.json')
            _write(sealed, value)
            for name in ('PUBLISHING', 'PUBLISHED'):
                path = run / f'FORMAL_EVALUATION_{name}.json'
                value = json.loads(path.read_text())
                value['transaction_sha256'] = 'd' * 64
                value['seal']['sha256'] = _sha(sealed)
                _write(path, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_wrong_isolet_vector_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory), dataset='isolet')
            for name in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = job / name
                value = json.loads(path.read_text())
                command = value['command']
                command[command.index('--vector_npz') + 1] = '/wrong/cache.npz'
                _write(path, value)
            config = job / 'run' / 'config.json'
            value = json.loads(config.read_text())
            value['vector_npz'] = '/wrong/cache.npz'
            _write(config, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_changed_reported_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            success = job / 'JOB_SUCCESS.json'
            value = json.loads(success.read_text())
            value['metrics']['aa_final_cil'] += 0.1
            _write(success, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_two_party_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            config = job / 'run' / 'config.json'
            value = json.loads(config.read_text())
            value['num_parties'] = 2
            _write(config, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_wrong_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            success = job / 'JOB_SUCCESS.json'
            value = json.loads(success.read_text())
            value['seed'] = 43
            _write(success, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_changed_training_epochs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            for filename in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = job / filename
                value = json.loads(path.read_text())
                command = value['command']
                command[command.index('--epochs_per_task') + 1] = '1'
                _write(path, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_changed_saved_training_epochs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            config = job / 'run' / 'config.json'
            value = json.loads(config.read_text())
            value['epochs_per_task'] = 1
            _write(config, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_unrelated_entrypoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            for filename in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = job / filename
                value = json.loads(path.read_text())
                value['command'][1] = '/unrelated/main.py'
                _write(path, value)
            with self.assertRaises(ValueError):
                verify_local_job(job)

    def test_archive_manifest_accepts_exact_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            manifest = _archive_manifest(job)
            self.assertEqual(
                verify_archive_manifest(job, manifest)['job'],
                'cifar100/mir/42')

    def test_archive_manifest_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            manifest = _archive_manifest(job)
            checkpoint = job / 'run' / 'checkpoints' / 'formal_final.pt'
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_archive_manifest(job, manifest)

    def test_unknown_legacy_tree_algorithm_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            manifest = _archive_manifest(job)
            value = json.loads(manifest.read_text())
            value['tree_sha256'] = '0' * 64
            del value['source_tree_sha256']
            del value['destination_tree_sha256']
            _write(manifest, value)
            with self.assertRaises(InconclusiveEvidence):
                verify_archive_manifest(job, manifest)

    def test_unrelated_parent_tree_cannot_attest_a_job(self):
        with tempfile.TemporaryDirectory() as directory:
            with tempfile.TemporaryDirectory() as other:
                root = Path(directory)
                job = _job(root)
                _archive_manifest(job)
                rows = []
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        rows.append([
                            path.relative_to(root).as_posix(),
                            path.stat().st_size, _sha(path),
                        ])
                digest = hashlib.sha256(json.dumps(
                    rows, ensure_ascii=False,
                    separators=(',', ':')).encode()).hexdigest()
                manifest = Path(other) / 'manifest.json'
                _write(manifest, {
                    'job': 'cifar100/mir/42', 'code_commit': COMMIT,
                    'source': '/original', 'destination': str(root),
                    'files': len(rows),
                    'bytes': sum(row[1] for row in rows),
                    'source_tree_sha256': digest,
                    'destination_tree_sha256': digest,
                })
                with self.assertRaises(ValueError):
                    verify_archive_manifest(job, manifest)


if __name__ == '__main__':
    unittest.main()
