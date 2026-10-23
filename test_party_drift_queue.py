import tempfile
import unittest
import json
import os
from pathlib import Path

import party_drift_queue as queue


class QueuePolicyTests(unittest.TestCase):
    def test_queue_run_lock_excludes_concurrent_recovery(self):
        self.assertTrue(hasattr(queue, 'queue_run_lock'))
        if os.name != 'posix':
            return
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            with queue.queue_run_lock(root):
                with self.assertRaises(RuntimeError):
                    with queue.queue_run_lock(root):
                        pass

    def test_recovery_accounts_for_plan_written_before_start_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = 'isolet/drift/0.5/70'
            job = root / key
            job.mkdir(parents=True)
            (job / 'attempt-01.plan.json').write_text(json.dumps({
                'attempt': 1, 'job_key': key}))
            result = queue.recover_interrupted_attempts(
                root, 'm', 'power cut before start', {key: 'a' * 64})
            self.assertEqual(result['retry_attempts'], {key: 2})
            history = [json.loads(line) for line in
                       (root / 'ATTEMPTS.jsonl').read_text().splitlines()]
            self.assertEqual([event['kind'] for event in history],
                             ['start', 'finish'])

    def test_recovery_keeps_prior_resume_record_after_second_interruption(self):
        self.assertTrue(hasattr(queue, 'record_resume'))
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            first = queue.record_resume(job, 2, 'isolet/drift/0.5/70',
                                        'm', 'first reboot')
            original = first.read_bytes()
            queue.record_resume(job, 2, 'isolet/drift/0.5/70',
                                'm', 'second reboot')
            self.assertEqual(first.read_bytes(), original)
            with self.assertRaises(ValueError):
                queue.record_resume(job, 2, 'isolet/drift/0.5/71',
                                    'm', 'different job')

    def test_recovery_detects_orphaned_worker_before_retry(self):
        self.assertTrue(hasattr(queue, 'running_worker_keys'))
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            (proc / '123').mkdir()
            (proc / '123' / 'cmdline').write_bytes(
                b'python\0run_party_drift_development.py\0--manifest\0'
                b'/tmp/manifest.jsonl\0--job-key\0isolet/drift/0.5/70\0')
            self.assertEqual(queue.running_worker_keys(
                '/tmp/manifest.jsonl', proc), {'isolet/drift/0.5/70'})

    def test_resumed_queue_uses_new_append_only_lock(self):
        self.assertTrue(hasattr(queue, 'active_queue_lock'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'QUEUE_LOCK.json').write_text(json.dumps({'pid': 1}))
            self.assertEqual(queue.active_queue_lock(root)['pid'], 1)
            (root / 'QUEUE_LOCK_RESUME_01.json').write_text(json.dumps({'pid': 2}))
            self.assertEqual(queue.active_queue_lock(root)['pid'], 2)

    def test_interrupted_queue_preserves_attempts_and_only_retries_infrastructure(self):
        self.assertTrue(hasattr(queue, 'recover_interrupted_attempts'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infra = 'isolet/drift/0.5/70'
            anomaly = 'isolet/drift/0.5/71'
            for key, log in ((infra, 'host lost power'),
                             (anomaly, 'FloatingPointError: non-finite loss')):
                job = root / key
                job.mkdir(parents=True)
                (job / 'attempt-01.log').write_text(log)
                (job / 'attempt-01.plan.json').write_text(
                    '{"attempt":1,"job_key":"' + key + '"}')
            events = root / 'ATTEMPTS.jsonl'
            events.write_text(''.join(json.dumps({
                'kind': 'start', 'job_key': key, 'attempt': 1,
                'config_sha256': 'a' * 64}) + '\n'
                for key in (infra, anomaly)))
            result = queue.recover_interrupted_attempts(
                root, 'm', 'host reboot', {infra: 'a' * 64,
                                          anomaly: 'a' * 64})
            self.assertEqual(result['retry_attempts'], {infra: 2})
            self.assertEqual(result['outcomes'][anomaly], 'training_anomaly')
            self.assertEqual(result['outcomes'][infra], 'infrastructure')
            self.assertEqual(len(events.read_text().splitlines()), 4)
            self.assertTrue((root / 'QUEUE_RECOVERY_01.json').is_file())
            again = queue.recover_interrupted_attempts(
                root, 'm', 'second host reboot', {infra: 'a' * 64,
                                                 anomaly: 'a' * 64})
            self.assertEqual(again['retry_attempts'], {infra: 2})
            self.assertEqual(len(events.read_text().splitlines()), 4)
            self.assertTrue((root / 'QUEUE_RECOVERY_02.json').is_file())

    def test_retry_requires_previous_infrastructure_failure_and_reason(self):
        self.assertTrue(hasattr(queue, 'retry_attempt'))
        key = 'upmc_food101/drift/0.5/70'
        failed = [{'kind': 'finish', 'job_key': key, 'attempt': 1,
                   'classification': 'infrastructure'}]
        self.assertEqual(queue.retry_attempt(failed, key, 'host reboot'), 2)
        with self.assertRaises(ValueError):
            queue.retry_attempt(failed, key, '')
        failed[0]['classification'] = 'training_anomaly'
        with self.assertRaises(ValueError):
            queue.retry_attempt(failed, key, 'retry anyway')

    def test_gpu_gate_requires_both_low_memory_and_low_utilization(self):
        self.assertTrue(hasattr(queue, 'gpu_state_is_idle'))
        self.assertTrue(queue.gpu_state_is_idle('1, 0'))
        self.assertFalse(queue.gpu_state_is_idle('4096, 0'))
        self.assertFalse(queue.gpu_state_is_idle('1, 90'))
        with self.assertRaises(ValueError):
            queue.gpu_state_is_idle('malformed')

    def test_failure_classification_does_not_retry_training_anomaly(self):
        self.assertTrue(hasattr(queue, 'classify_failure'))
        self.assertEqual(queue.classify_failure(1, 'FloatingPointError: non-finite loss'),
                         'training_anomaly')
        self.assertEqual(queue.classify_failure(1, 'torch.cuda.OutOfMemoryError'),
                         'infrastructure')
        self.assertEqual(queue.classify_failure(-9, ''), 'infrastructure')
        self.assertEqual(queue.classify_failure(1, 'unknown traceback'), 'unknown')

    def test_order_starts_upmc_and_attempt_plan_is_exclusive(self):
        self.assertTrue(hasattr(queue, 'ordered_jobs'))
        self.assertTrue(hasattr(queue, 'reserve_attempt'))
        rows = [
            {'job_key': 'cifar100/drift/0.5/70', 'dataset': 'cifar100'},
            {'job_key': 'isolet/drift/0.5/70', 'dataset': 'isolet'},
            {'job_key': 'upmc_food101/drift/0.5/70', 'dataset': 'upmc_food101'},
        ]
        self.assertEqual([row['dataset'] for row in queue.ordered_jobs(rows)],
                         ['upmc_food101', 'isolet', 'cifar100'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'one-job'
            queue.reserve_attempt(root, 1, rows[2]['job_key'])
            with self.assertRaises(FileExistsError):
                queue.reserve_attempt(root, 1, rows[2]['job_key'])
            queue.reserve_attempt(root, 2, rows[2]['job_key'])


if __name__ == '__main__':
    unittest.main()
