import tempfile
import unittest
from pathlib import Path

import party_drift_queue as queue


class QueuePolicyTests(unittest.TestCase):
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
