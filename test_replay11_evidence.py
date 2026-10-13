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


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(root):
    job = root / 'cifar100' / 'mir' / '42'
    run = job / 'run'
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'frozen-final-checkpoint')
    history = [
        {
            'step': f'event_{i}_CIL',
            'per_task_accs': (
                {f'task_{j}': 0.5 - 0.01 * j for j in range(10)}
                if i == 9 else {f'task_{i}': 0.8}
            ),
            **({'per_task_accs_taskil': {
                f'task_{j}': 0.7 for j in range(10)
            }} if i == 9 else {}),
        }
        for i in range(10)
    ]
    frozen_snapshots = []
    for i in range(10):
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
        'cl_method': 'mir', 'seed': 42, 'num_tasks': 10,
        'num_parties': 4, 'formal_deferred_evaluation': True,
    }
    result = {
        'config': dict(config), 'task_acc_history': history,
        'source_provenance': {'source_commit': COMMIT},
    }
    _write(run / 'config.json', config)
    _write(run / 'results.json', result)
    _write(run / 'FORMAL_STATE_FROZEN.json', {
        'schema_version': 1, 'status': 'frozen',
        'identity': {
            'source_commit': COMMIT, 'snapshots': frozen_snapshots,
        },
    })
    _write(run / 'FORMAL_EVALUATION_PUBLISHED.json', {
        'schema_version': 1, 'status': 'published',
        'source_provenance': {'source_commit': COMMIT},
        'checkpoint': {'path': 'checkpoints/formal_final.pt',
                       'sha256': _sha(checkpoint)},
        'results': {'path': 'results.json',
                    'sha256': _sha(run / 'results.json')},
    })
    _write(run / 'formal_access' / 'test.consumed.json', {
        'schema_version': 1, 'status': 'consumed',
        'access': {'split': 'test', 'event_idx': 9,
                   'phase': 'final_test_post_install'},
    })
    command = matrix.command_for(
        ('cifar100', 'mir', 42), job,
        python=Path('python'), worktree=Path('/tmp/code'),
        device='cuda:0', data_root=root / 'data')
    plan = {
        'dataset': 'cifar100', 'method': 'mir', 'seed': 42,
        'smoke': False, 'code_commit': COMMIT, 'command': command,
        'data_hashes': matrix.expected_data('cifar100'),
    }
    _write(job / 'JOB_PLAN.json', plan)
    _write(job / 'JOB_SUCCESS.json', {
        **plan,
        'metrics': matrix.metrics_from_history(
            history, expected_tasks=10, deferred=True),
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
            _write(published, marker)
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


if __name__ == '__main__':
    unittest.main()
