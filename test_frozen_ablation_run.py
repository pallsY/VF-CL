import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import frozen_ablation_run as queue
from frozen_ablation_manifest import (BASE_COMMIT, DATASETS, PATCH_COMMIT,
                                       _pretty_bytes, patch_diff_sha256)
from frozen_ablation_run import (gpu_eligible, verify_formal_seal,
                                 verify_preflight, fail_stop_peers)


class AblationRunTests(unittest.TestCase):
    def test_nonzero_exit_immediately_stops_peer_process_group(self):
        stop = __import__('threading').Event()
        class Process:
            pid = 12345
            def poll(self):
                return None
        with mock.patch('frozen_ablation_run.os.killpg', create=True) as killpg:
            self.assertTrue(fail_stop_peers({1: Process()}, 0, stop))
        self.assertTrue(stop.is_set())
        killpg.assert_called_once_with(12345, queue.signal.SIGTERM)

    def test_queue_preflight_requires_smoke_and_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifests/manifest.jsonl'
            manifest.parent.mkdir()
            with self.assertRaisesRegex(ValueError, 'preflight'):
                verify_preflight(manifest, [{'preflight_sha256': 'a' * 64}])

    def test_queue_preflight_accepts_only_hash_bound_smoke_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifests/manifest.jsonl'
            evidence = manifest.parent / 'preflight'
            evidence.mkdir(parents=True)
            report = {
                'default_on_bitwise': True,
                'off_task0_identical': True,
                'sample_order_identical_all_tasks': True,
                'fim_tensor_values_compared': True,
                'off_task1_model_diverged': True,
                'off_applied_freeze_fraction': 0,
                'test_accessed': False,
                'orchestration_commit': 'a' * 40,
                'smoke_script_sha256': 'b' * 64,
                'raw_smoke_sha256': {mode: 'c' * 64 for mode in
                                     ('base', 'default_on', 'off')},
                'raw_config_sha256': {mode: 'd' * 64 for mode in
                                      ('base', 'default_on', 'off')},
                'raw_no_test_sha256': {mode: 'e' * 64 for mode in
                                       ('base', 'default_on', 'off')},
            }
            smoke_bytes = _pretty_bytes({dataset: report for dataset in DATASETS})
            smoke_path = evidence / 'SMOKE_COMPARISON.json'
            smoke_path.write_bytes(smoke_bytes)
            diff_sha = patch_diff_sha256()
            review = {'reviewer_id': 'independent-reviewer',
                      'verdict': 'approved', 'base_commit': BASE_COMMIT,
                      'patch_commit': PATCH_COMMIT,
                      'patch_diff_sha256': diff_sha,
                      'blocking_findings': []}
            review_bytes = _pretty_bytes(review)
            review_path = evidence / 'PATCH_REVIEW.json'
            review_path.write_bytes(review_bytes)
            record = {'schema_version': 1, 'status': 'approved',
                      'publication_preflight_sha256': 'f' * 64,
                      'reviewer_id': review['reviewer_id'],
                      'base_commit': BASE_COMMIT,
                      'patch_commit': PATCH_COMMIT,
                      'patch_diff_sha256': diff_sha,
                      'smoke_comparison_sha256': hashlib.sha256(
                          smoke_bytes).hexdigest(),
                      'patch_review_sha256': hashlib.sha256(
                          review_bytes).hexdigest()}
            record_bytes = _pretty_bytes(record)
            (evidence / 'PRELAUNCH_PREFLIGHT.json').write_bytes(record_bytes)
            row = {
                'publication_preflight_sha256': 'f' * 64,
                'preflight_sha256': hashlib.sha256(record_bytes).hexdigest(),
                'smoke_comparison_sha256': hashlib.sha256(smoke_bytes).hexdigest(),
                'patch_review_sha256': hashlib.sha256(review_bytes).hexdigest(),
                'patch_diff_sha256': diff_sha,
            }
            with mock.patch('frozen_ablation_smoke.validate_smoke_records'), mock.patch.object(queue, 'validate_publication_preflight', return_value='f' * 64):
                self.assertEqual(verify_preflight(manifest, [row])['status'],
                                 'approved')
            smoke_path.write_bytes(smoke_bytes + b' ')
            with mock.patch('frozen_ablation_smoke.validate_smoke_records'), mock.patch.object(queue, 'validate_publication_preflight', return_value='f' * 64):
                with self.assertRaisesRegex(ValueError, 'preflight'):
                    verify_preflight(manifest, [row])

    def test_gpu_admission_requires_correct_physical_card_and_headroom(self):
        good = {'index': 0,
                'uuid': 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
                'name': 'NVIDIA GeForce RTX 3080', 'free_mib': 3072}
        self.assertTrue(gpu_eligible(good, 0))
        self.assertFalse(gpu_eligible({**good, 'free_mib': 3071}, 0))
        self.assertFalse(gpu_eligible({**good, 'uuid': 'wrong'}, 0))

    def producer_fixture(self):
        import adaptive_consolidation_audit as producer
        from metrics import MetricsTracker
        if __import__('os').name == 'nt':
            # Production dir_fd I/O is Linux-only; preserve the real serializer
            # and publication logic while adapting temporary-file I/O on Windows.
            for name, replacement in (
                ('_safe_open', lambda path, mode='rb': Path(path).open(mode)),
                ('atomic_write_new_json', lambda path, value: Path(path).write_bytes(
                    producer._strict_json(value).encode('utf-8'))),
            ):
                patch = mock.patch.object(producer, name, replacement)
                patch.start()
                self.addCleanup(patch.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name)
        def write(name, value):
            producer.atomic_write_new_json(run / name, value)
        digest = lambda value: hashlib.sha256(producer._strict_json(value).encode()).hexdigest()
        protocol = {'num_tasks': 1}
        identity = {'protocol': protocol, 'protocol_sha256': digest(protocol),
                    'source_provenance': {'fixture': 'synthetic publisher'},
                    'task_classes': {'0': [0, 1]}}
        freeze = {'fixture': 'frozen'}
        transaction = producer._formal_transaction_sha256(identity, freeze)
        common = dict(schema_version=1, transaction_sha256=transaction,
                      identity=identity, freeze=freeze)
        pending = dict(common, status='pending')
        consuming = dict(common, status='consuming', cache_identity={})
        evaluation = MetricsTracker().to_dict()
        complete = dict(common, status='complete', cache_identity={},
                        evaluation=evaluation, evaluation_sha256=digest(evaluation))
        write('FORMAL_EVALUATION_PENDING.json', pending)
        write('FORMAL_EVALUATION_CONSUMING.json', consuming)
        write('FORMAL_EVALUATION_COMPLETE.json', complete)
        write('FORMAL_EVALUATION_SEALED.json', producer._formal_seal_payload(
            run, identity, freeze, pending, consuming, complete))
        final = dict(evaluation, source_provenance=identity['source_provenance'])
        producer.begin_formal_deferred_publication(
            output_dir=run, tracker_state=evaluation, final_result=final)
        (run / 'checkpoints').mkdir()
        checkpoint = run / 'checkpoints/formal_final.pt'
        checkpoint.write_bytes(b'synthetic model; no training')
        results = run / 'results.json'
        # Match runner's JSON writer, which differs from _strict_json.
        results.write_text(json.dumps(final, indent=2), encoding='utf-8')
        record = producer._formal_file_record(checkpoint, run)
        with mock.patch.object(producer, '_validate_formal_publication_checkpoint',
                               return_value=({}, record)), mock.patch.object(
                producer, '_revalidate_formal_publication_inputs'):
            producer.complete_formal_deferred_publication(
                args=SimpleNamespace(), output_dir=run,
                final_checkpoint=checkpoint, results_path=results)
        (run / 'formal_access').mkdir()
        access = dict(event='first_iteration', split='test',
                      phase='final_test_post_install', event_idx=0, task_id=0,
                      timeline_step='event_0_CIL', classes=[0, 1],
                      loader_key=repr(('test', (0, 1))))
        write('formal_access/test.consumed.json',
              dict(schema_version=1, status='consumed', access=access))
        (run / 'data_flow_audit.jsonl').write_text(json.dumps(access) + '\n')
        return SimpleNamespace(run=run, results=results)

    @unittest.skipIf(__import__('os').name == 'nt', 'producer fixture requires Linux dir_fd')
    def test_full_linux_producer_fixture_passes_formal_seal(self):
        from test_three_dataset_formal_audit import RealProducerFixture
        fixture = RealProducerFixture()
        self.addCleanup(fixture.close)
        sealed = verify_formal_seal(fixture.run)
        self.assertNotEqual(sealed['results_file_sha256'],
                            sealed['results_content_sha256'])

    def test_formal_seal_accepts_real_producer_json_and_names_both_digests(self):
        from adaptive_consolidation_audit import _strict_json
        fixture = self.producer_fixture()
        raw = fixture.results.read_bytes()
        canonical = _strict_json(json.loads(raw)).encode('utf-8')
        self.assertNotEqual(raw, canonical)
        result = verify_formal_seal(fixture.run)
        self.assertEqual(result['results_file_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result['results_content_sha256'], hashlib.sha256(canonical).hexdigest())

    def test_formal_seal_rejects_content_change_even_with_updated_file_record(self):
        import adaptive_consolidation_audit as producer
        fixture = self.producer_fixture()
        value = json.loads(fixture.results.read_text())
        value['unexpected_metadata'] = 'changed after publication'
        fixture.results.write_text(json.dumps(value))
        published_path = fixture.run / 'FORMAL_EVALUATION_PUBLISHED.json'
        published = json.loads(published_path.read_text())
        published['results'] = producer._formal_file_record(fixture.results, fixture.run)
        published_path.chmod(0o600)
        published_path.write_text(json.dumps(published))
        with self.assertRaisesRegex(ValueError, 'publication content or file record'):
            verify_formal_seal(fixture.run)

    def test_formal_seal_rejects_tampering_and_duplicate_access(self):
        for target in ('results.json', 'checkpoints/formal_final.pt',
                       'FORMAL_EVALUATION_SEALED.json',
                       'FORMAL_EVALUATION_COMPLETE.json',
                       'FORMAL_EVALUATION_PUBLISHED.json',
                       'FORMAL_EVALUATION_PUBLISHING.json',
                       'FORMAL_EVALUATION_PENDING.json',
                       'FORMAL_EVALUATION_CONSUMING.json',
                       'formal_access/test.consumed.json',
                       'duplicate_history', 'duplicate'):
            with self.subTest(target=target):
                fixture = self.producer_fixture()
                if target == 'duplicate':
                    (fixture.run / 'formal_access/test.extra.json').write_text('{}')
                elif target == 'duplicate_history':
                    path = fixture.run / 'data_flow_audit.jsonl'
                    path.write_bytes(path.read_bytes() * 2)
                else:
                    path = fixture.run / target
                    path.chmod(0o600)
                    if target in ('FORMAL_EVALUATION_PUBLISHED.json',
                                  'FORMAL_EVALUATION_PUBLISHING.json',
                                  'FORMAL_EVALUATION_PENDING.json',
                                  'FORMAL_EVALUATION_CONSUMING.json'):
                        value = json.loads(path.read_text())
                        value['transaction_sha256'] = '0' * 64
                        path.write_text(json.dumps(value))
                    elif target == 'formal_access/test.consumed.json':
                        value = json.loads(path.read_text())
                        value['access']['classes'] = [1, 0]
                        path.write_text(json.dumps(value))
                    elif target == 'results.json':
                        value = json.loads(path.read_text())
                        value['source_provenance'] = {'tampered': True}
                        path.write_text(json.dumps(value))
                    else:
                        path.write_bytes(path.read_bytes() + b' ')
                with self.assertRaises(ValueError):
                    verify_formal_seal(fixture.run)

    def test_process_launch_failure_is_sealed_and_stops_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'job_key': 'isolet/fim_off/42', 'arm': 'fim_off',
                   'physical_gpu': 0, 'config_sha256': 'a' * 64,
                   'code_commit': 'b' * 40,
                   'orchestration_commit': 'c' * 40}
            config = {'seed': 42, 'num_tasks': 2}
            job = (row, config, root / 'config.json', root / 'fim_off/isolet/42')
            snapshot = {'index': 0,
                        'uuid': queue.GPU_UUID[0],
                        'name': 'NVIDIA GeForce RTX 3080',
                        'free_mib': 9000}
            with mock.patch.object(queue, 'RESULT_ROOT', root), \
                 mock.patch.object(queue, 'load_jobs', return_value=[job]), \
                 mock.patch.object(queue.shutil, 'disk_usage',
                                   return_value=SimpleNamespace(free=100 << 30)), \
                 mock.patch.object(queue, 'wait_headroom', return_value=snapshot), \
                 mock.patch.object(queue, 'gpu_snapshot', return_value=snapshot), \
                 mock.patch.object(queue.subprocess, 'Popen', side_effect=OSError('launch failed')):
                with self.assertRaisesRegex(RuntimeError, 'stopped'):
                    queue.run_queue(root / 'manifest.jsonl', 'd' * 64,
                                    {'fim_off': root, 'no_partykd': root})
            events = [json.loads(line) for line in
                      (root / 'ATTEMPTS.jsonl').read_text().splitlines()]
            self.assertEqual([event['kind'] for event in events],
                             ['start', 'finish'])
            self.assertEqual(events[-1]['classification'],
                             'infrastructure_candidate')
            self.assertTrue((root / 'QUEUE_FAILURE.json').is_file())


if __name__ == '__main__':
    unittest.main()
