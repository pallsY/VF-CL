import json
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import replay11_matrix as matrix
from test_replay11_evidence import (
    _archive_manifest, _canonical_sha, _job, _sha, _write,
)
from adaptive_consolidation_audit import FORMAL_SOURCE_FILES
from replay11_evidence import (
    CORRECTED_COMMIT, PRODUCER_COMMIT, _merge_verified_attestations,
    equivalence_certificate,
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
            python=Path('python'), worktree=Path('/tmp/vfcl-replay11-paper-v3-full'),
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
    identity['task_classes'] = (
        {str(i): [int(value) for value in group.split(',')]
         for i, group in enumerate(config['custom_tasks'].split('|'))}
        if config.get('custom_tasks') else
        {str(i): list(range(i * config['classes_per_task'],
                            (i + 1) * config['classes_per_task']))
         for i in range(config['num_tasks'])})
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


def _vector_training_flow(root):
    from data_utils import TaskManager, VFLDataset
    from determinism import tensor_sha256

    data_root = root / 'data'
    isolet = data_root / 'isolet'
    isolet.mkdir(parents=True)
    np.savez(isolet / 'isolet_vfl.npz',
             X=np.arange(40, dtype=np.float32).reshape(10, 4),
             y=np.array([0] * 5 + [1] * 5),
             train_idx=np.arange(8), test_idx=np.array([8, 9]),
             view_names=np.array(['a', 'b', 'c', 'd']),
             range_lo=np.array([0, 1, 2, 3]),
             range_hi=np.array([1, 2, 3, 4]))
    (isolet / 'isolet_vfl.metadata.json').write_text('{}')
    config = {
        'data': 'tabvfl', 'num_classes': 2, 'num_parties': 4,
        'num_tasks': 2, 'custom_tasks': '0|1', 'classes_per_task': 1,
        'seed': 42, 'batch_size': 2, 'num_workers': 0,
        'deterministic': 1, 'data_flow_audit': 1,
        'epochs_per_task': 2, 'replay_mode': 'class',
        'bic_enabled': 0, 'lambda_validation_enabled': 0,
        'data_path': str(data_root),
        'vector_npz': str(isolet / 'isolet_vfl.npz'),
        'output_dir': str(root / 'scratch'),
    }
    dataset = VFLDataset(SimpleNamespace(**config))
    tasks = TaskManager(SimpleNamespace(**config))
    rows = []
    for task_id in range(2):
        tasks.advance_task(task_id)
        loader = dataset.get_train_loader(tasks.task_classes[task_id])
        for epoch in range(2):
            for batch, (batch_x, _batch_y) in enumerate(loader):
                order = loader.audit_sampler.epoch_orders[-1]
                start = batch * loader.batch_size
                rows.append({
                    'loader_key': loader.audit_key,
                    'loader_iteration': epoch, 'batch': batch,
                    'indices': order[start:start + len(batch_x)],
                    'dtype': str(batch_x.dtype),
                    'shape': list(batch_x.shape),
                    'batch_sha256': tensor_sha256(batch_x),
                })
    rows.append({'split': 'test'})
    run = root / 'run'
    run.mkdir()
    (run / 'data_flow_audit.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return run, config, data_root, rows


class Replay11V3Test(unittest.TestCase):
    def test_vector_flow_rejects_truncation_and_changed_batch_sha(self):
        from replay11_evidence import _verify_training_flow_strict
        with tempfile.TemporaryDirectory() as directory:
            run, config, data_root, rows = _vector_training_flow(Path(directory))
            self.assertTrue(_verify_training_flow_strict(
                run, config, data_root, 2)['batch_bytes_replayed'])
            path = run / 'data_flow_audit.jsonl'
            for changed in (rows[:3] + rows[4:],
                            [{**rows[0], 'batch_sha256': '0' * 64}]
                            + rows[1:]):
                path.write_text(''.join(json.dumps(row) + '\n'
                                        for row in changed), encoding='utf-8')
                with self.assertRaisesRegex(ValueError,
                                            'training data-flow reconstruction'):
                    _verify_training_flow_strict(run, config, data_root, 2)

    def test_pinned_locator_rejects_copy_with_rewritten_plan(self):
        from replay11_evidence import verify_pinned_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _corrected_job(root / 'official')
            copy = root / 'copy' / 'cifar100' / 'mir' / '42'
            shutil.copytree(original, copy)
            for name in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = copy / name
                value = json.loads(path.read_text())
                command = value['command']
                command[command.index('--results_dir') + 1] = str(copy)
                _write(path, value)
            with self.assertRaisesRegex(ValueError, 'pinned result locator'):
                verify_pinned_job(
                    copy, producer_commit=CORRECTED_COMMIT,
                    results_root=root / 'official', data_root=root / 'data')

    def test_pinned_data_root_must_match_training_command(self):
        from replay11_evidence import verify_pinned_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = _corrected_job(root / 'official')
            with self.assertRaisesRegex(ValueError, 'pinned data root'):
                verify_pinned_job(
                    job, producer_commit=CORRECTED_COMMIT,
                    results_root=root / 'official', data_root=root / 'other')

    def test_pinned_job_rejects_synthetic_one_batch_per_task(self):
        import replay11_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = _corrected_job(root / 'official')
            with patch.object(evidence, '_verify_authoritative_data',
                              return_value=matrix.expected_data('cifar100')):
                with self.assertRaisesRegex(ValueError,
                                            'training data-flow reconstruction'):
                    evidence.verify_pinned_job(
                        job, producer_commit=CORRECTED_COMMIT,
                        results_root=root / 'official',
                        data_root=root / 'official' / 'data')

    def test_strict_checkpoint_rejects_opaque_bytes(self):
        from replay11_evidence import _verify_checkpoint_method_state
        with tempfile.TemporaryDirectory() as directory:
            job = _corrected_job(Path(directory))
            config = json.loads((job / 'run' / 'config.json').read_text())
            with self.assertRaisesRegex(ValueError, 'checkpoint method state'):
                _verify_checkpoint_method_state(
                    job / 'run' / 'checkpoints' / 'formal_final.pt',
                    config, 10)

    def test_frozen_task_classes_must_match_complete_config(self):
        from replay11_evidence import _verify_frozen_task_classes
        with tempfile.TemporaryDirectory() as directory:
            job = _corrected_job(Path(directory))
            run = job / 'run'
            config = json.loads((run / 'config.json').read_text())
            frozen = json.loads((run / 'FORMAL_STATE_FROZEN.json').read_text())
            classes = {str(i): list(range(i * 10, (i + 1) * 10))
                       for i in range(10)}
            frozen['identity']['task_classes'] = classes
            self.assertEqual(_verify_frozen_task_classes(frozen, config, 10),
                             classes)
            frozen['identity']['task_classes']['9'] = list(range(90, 99))
            with self.assertRaisesRegex(ValueError, 'frozen task classes'):
                _verify_frozen_task_classes(frozen, config, 10)
            frozen['identity']['task_classes']['9'] = list(range(90, 100)) + [0]
            with self.assertRaisesRegex(ValueError, 'frozen task classes'):
                _verify_frozen_task_classes(frozen, config, 10)

    def test_frozen_state_hashes_reject_resealed_checkpoint_mutations(self):
        from adaptive_consolidation_audit import trainer_state_sha256
        from replay11_evidence import _verify_frozen_checkpoint_state
        payload = {
            'trainer_state': {'weight': torch.ones(2)},
            'cl_state': {'data': torch.ones(2)},
            'rng_state': {'torch': torch.ones(2, dtype=torch.uint8)},
            'tracker_state': {'history': [1]},
        }
        frozen = {
            'trainer_state_sha256': trainer_state_sha256(payload['trainer_state']),
            'method_state_sha256': trainer_state_sha256(payload['cl_state']),
            'rng_state_sha256': trainer_state_sha256(payload['rng_state']),
        }
        evaluation = dict(payload['tracker_state'])
        _verify_frozen_checkpoint_state(payload, frozen, evaluation)
        for field in ('trainer_state', 'cl_state', 'rng_state'):
            changed = dict(payload)
            changed[field] = {'changed': torch.ones(2)}
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, 'frozen checkpoint state'):
                _verify_frozen_checkpoint_state(changed, frozen, evaluation)
        with self.assertRaisesRegex(ValueError, 'frozen checkpoint state'):
            _verify_frozen_checkpoint_state(payload, frozen,
                                            {'history': [2]})

    def test_strict_checkpoint_rejects_wrong_method_state(self):
        from replay11_evidence import (_expected_protocol,
                                       _verify_checkpoint_method_state)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = _corrected_job(root)
            run = job / 'run'
            config = json.loads((run / 'config.json').read_text())
            provenance = json.loads((run / 'results.json').read_text())[
                'source_provenance']
            classes = ([[int(item) for item in group.split(',')]
                        for group in config['custom_tasks'].split('|')]
                       if config.get('custom_tasks') else
                       [list(range(i * config['classes_per_task'],
                                   (i + 1) * config['classes_per_task']))
                        for i in range(10)])
            checkpoint = {
                'schema_version': 4, 'task_id': 9, 'event_idx': 9,
                'step': 'event_9_CIL',
                'seen_task_classes': dict(enumerate(classes)),
                'protocol': _expected_protocol(config, provenance),
                'trainer_state': {
                    'bottoms': [{'weight': torch.ones(1)} for _ in range(4)],
                    'top_model': {'weight': torch.ones(1)},
                },
                'cl_state': {
                    'per_class_size': 20,
                    'data': {0: torch.ones(1, 4)},
                    'seen_count': {0: 1},
                },
                'tracker_state': {},
            }
            path = root / 'state.pt'
            torch.save(checkpoint, path)
            with patch('runner._valid_tracker_checkpoint_state',
                       return_value=True), patch(
                           'adaptive_consolidation_audit._safe_torch_load',
                           side_effect=lambda target: torch.load(
                               target, map_location='cpu', weights_only=True)):
                self.assertEqual(_verify_checkpoint_method_state(
                    path, config, 10)['method'], 'mir')
                checkpoint['cl_state'] = {'gaussians': {}, 'adapters': []}
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError,
                                            'checkpoint method state'):
                    _verify_checkpoint_method_state(path, config, 10)

    def test_corrected_full_transaction_attests_real_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            record = verify_local_job(_corrected_job(Path(directory)))
            self.assertEqual(record['code_commit'], CORRECTED_COMMIT)
            self.assertEqual(record['admission'], 'EVIDENCE_INCOMPLETE')
            self.assertEqual(record['source_sha256']['cl_methods/mir.py'],
                             __import__('replay11_evidence')._producer_source_sha256(
                                 CORRECTED_COMMIT)['cl_methods/mir.py'])
            self.assertEqual(len(record['transaction_sha256']), 64)
            self.assertEqual(record['data_hashes'],
                             matrix.expected_data('cifar100'))
            self.assertEqual(len(record['command_sha256']), 64)

    def test_authoritative_data_is_hashed_from_files(self):
        from replay11_evidence import _verify_authoritative_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sample.bin').write_bytes(b'authoritative')
            expected = {'sample.bin': hashlib.sha256(b'authoritative').hexdigest()}
            self.assertEqual(_verify_authoritative_data(root, expected), expected)
            (root / 'sample.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'data hash'):
                _verify_authoritative_data(root, expected)

    def test_catalog_rejects_99_unbound_rows(self):
        from test_replay11_evidence import _all_attestations
        with self.assertRaisesRegex(ValueError, 'job path'):
            merge_attestations(_all_attestations())

    def test_legacy_archive_without_original_is_unproven(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            manifest = _archive_manifest(job)
            record = verify_archive_manifest(job, manifest)
            self.assertEqual(record['status'], 'SOURCE_EQUIVALENCE_UNPROVEN')

    def test_copied_job_cannot_be_labelled_original(self):
        from replay11_evidence import _is_planned_original
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = _job(root / 'source')
            copy = root / 'copy' / 'cifar100' / 'mir' / '42'
            shutil.copytree(job, copy)
            self.assertTrue(_is_planned_original(job))
            self.assertFalse(_is_planned_original(copy))

    def test_archive_with_accessible_identical_original_is_proven(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _job(root / 'original')
            copy = root / 'copy' / 'cifar100' / 'mir' / '42'
            shutil.copytree(original, copy)
            manifest = _archive_manifest(copy)
            value = json.loads(manifest.read_text())
            value['source'] = str(original)
            _write(manifest, value)
            self.assertEqual(verify_archive_manifest(copy, manifest)['status'],
                             'SOURCE_EQUIVALENT')
            (original / 'training.log').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source and destination'):
                verify_archive_manifest(copy, manifest)

    def test_archive_source_must_be_the_planned_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _job(root / 'original')
            copy = root / 'copy' / 'cifar100' / 'mir' / '42'
            other = root / 'other' / 'cifar100' / 'mir' / '42'
            shutil.copytree(original, copy)
            shutil.copytree(original, other)
            manifest = _archive_manifest(copy)
            value = json.loads(manifest.read_text())
            value['source'] = str(other)
            _write(manifest, value)
            with self.assertRaisesRegex(ValueError, 'archive source path'):
                verify_archive_manifest(copy, manifest)

    def test_copied_corrected_job_needs_source_chain_for_admission(self):
        import replay11_evidence as evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _corrected_job(root / 'original')
            copy = root / 'copy' / 'cifar100' / 'mir' / '42'
            shutil.copytree(original, copy)
            with patch.object(evidence, '_verify_authoritative_data',
                              return_value=matrix.expected_data('cifar100')):
                record = verify_local_job(copy, data_root=root / 'data')
            self.assertEqual(record['admission'],
                             'SOURCE_EQUIVALENCE_UNPROVEN')
            manifest = _archive_manifest(copy)
            value = json.loads(manifest.read_text())
            value['source'] = str(original)
            value['code_commit'] = CORRECTED_COMMIT
            _write(manifest, value)
            with patch.object(evidence, '_verify_authoritative_data',
                              return_value=matrix.expected_data('cifar100')):
                proven = verify_local_job(
                    copy, data_root=root / 'data', archive_manifest=manifest)
            self.assertEqual(proven['admission'], 'formal')
            self.assertEqual(proven['archive_evidence']['status'],
                             'SOURCE_EQUIVALENT')

    def test_corrected_entrypoint_cannot_be_arbitrary_main(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _corrected_job(Path(directory))
            for name in ('JOB_PLAN.json', 'JOB_SUCCESS.json'):
                path = job / name
                value = json.loads(path.read_text())
                value['command'][1] = '/evil/main.py'
                _write(path, value)
            with self.assertRaisesRegex(ValueError, 'entrypoint'):
                verify_local_job(job)

    def test_complete_history_must_equal_published_results(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            run = job / 'run'
            complete_path = run / 'FORMAL_EVALUATION_COMPLETE.json'
            complete = json.loads(complete_path.read_text())
            complete['evaluation']['task_acc_history'][0]['per_task_accs']['task_0'] = .1
            complete['evaluation_sha256'] = _canonical_sha(complete['evaluation'])
            _write(complete_path, complete)
            sealed_path = run / 'FORMAL_EVALUATION_SEALED.json'
            sealed = json.loads(sealed_path.read_text())
            sealed['complete']['sha256'] = _sha(complete_path)
            _write(sealed_path, sealed)
            for name in ('PUBLISHING', 'PUBLISHED'):
                path = run / f'FORMAL_EVALUATION_{name}.json'
                value = json.loads(path.read_text())
                value['evaluation_sha256'] = complete['evaluation_sha256']
                value['seal']['sha256'] = _sha(sealed_path)
                _write(path, value)
            with self.assertRaisesRegex(ValueError, 'evaluation history'):
                verify_local_job(job)

    def test_missing_training_data_flow_is_not_basic_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            job = _job(Path(directory))
            flow = job / 'run' / 'data_flow_audit.jsonl'
            if flow.exists():
                flow.unlink()
            with self.assertRaisesRegex(ValueError, 'data-flow'):
                verify_local_job(job)

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
                if method == 'balanced_contrast':
                    self.assertIn('cl_methods/er.py',
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
            _merge_verified_attestations(rows)
        command = matrix.command_for(
            ('cifar100', 'er', 42), Path('/results'),
            python=Path('/python'), worktree=Path('/source'),
            device='cuda:0', data_root=Path('/data'))
        old['reuse_certificate'] = equivalence_certificate(
            ('cifar100', 'er', 42), command)
        self.assertEqual(_merge_verified_attestations(rows)['jobs'], 99)
        old['pilot_deferred'] = True
        with self.assertRaises(ValueError):
            _merge_verified_attestations(rows)


if __name__ == '__main__':
    unittest.main()
