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
                'preflight_sha256': hashlib.sha256(record_bytes).hexdigest(),
                'smoke_comparison_sha256': hashlib.sha256(smoke_bytes).hexdigest(),
                'patch_review_sha256': hashlib.sha256(review_bytes).hexdigest(),
                'patch_diff_sha256': diff_sha,
            }
            with mock.patch('frozen_ablation_smoke.validate_smoke_records'):
                self.assertEqual(verify_preflight(manifest, [row])['status'],
                                 'approved')
            smoke_path.write_bytes(smoke_bytes + b' ')
            with mock.patch('frozen_ablation_smoke.validate_smoke_records'):
                with self.assertRaisesRegex(ValueError, 'preflight'):
                    verify_preflight(manifest, [row])

    def test_gpu_admission_requires_correct_physical_card_and_headroom(self):
        good = {'index': 0,
                'uuid': 'GPU-6c7fee92-a6fe-5525-c554-bddb8e14943f',
                'name': 'NVIDIA GeForce RTX 3080', 'free_mib': 3072}
        self.assertTrue(gpu_eligible(good, 0))
        self.assertFalse(gpu_eligible({**good, 'free_mib': 3071}, 0))
        self.assertFalse(gpu_eligible({**good, 'uuid': 'wrong'}, 0))

    def test_formal_seal_requires_single_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'formal_access').mkdir()
            (run / 'checkpoints').mkdir()
            (run / 'results.json').write_text('{}')
            (run / 'checkpoints/formal_final.pt').write_bytes(b'model')
            (run / 'FORMAL_EVALUATION_COMPLETE.json').write_text(
                json.dumps({'status': 'complete'}))
            (run / 'FORMAL_EVALUATION_SEALED.json').write_text(
                json.dumps({'status': 'sealed'}))
            (run / 'FORMAL_EVALUATION_PUBLISHED.json').write_text(
                json.dumps({'status': 'published',
                    'results_sha256': hashlib.sha256(b'{}').hexdigest(),
                    'transaction_sha256': 'same'}))
            access = run / 'formal_access/test.consumed.json'
            access.write_text(json.dumps({'status': 'consumed',
                'access': {'split': 'test', 'event': 'first_iteration'}}))
            result = verify_formal_seal(run)
            self.assertEqual(len(result['results_sha256']), 64)
            access.unlink()
            with self.assertRaises(ValueError):
                verify_formal_seal(run)

    def test_formal_seal_rejects_changed_result_after_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'formal_access').mkdir()
            (run / 'checkpoints').mkdir()
            (run / 'results.json').write_text('{"changed": true}')
            (run / 'checkpoints/formal_final.pt').write_bytes(b'model')
            for name, value in (
                ('FORMAL_EVALUATION_COMPLETE.json', {'status': 'complete'}),
                ('FORMAL_EVALUATION_SEALED.json', {'status': 'sealed'}),
                ('FORMAL_EVALUATION_PUBLISHED.json', {
                    'status': 'published',
                    'results_sha256': hashlib.sha256(b'{}').hexdigest()}),
                ('formal_access/test.consumed.json', {'status': 'consumed',
                    'access': {'split': 'test', 'event': 'first_iteration'}}),
            ):
                (run / name).write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                verify_formal_seal(run)

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
