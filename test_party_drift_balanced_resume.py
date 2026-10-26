import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import party_drift_balanced_resume as balanced


def sealed_job(root, key, digest, attempt=1):
    run = root / key / f'attempt-{attempt:02d}'
    (run / 'checkpoints').mkdir(parents=True)
    checkpoint = run / 'checkpoints' / 'formal_final.pt'
    checkpoint.write_bytes(b'fixed checkpoint')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    flow = run / 'data_flow_audit.jsonl'
    flow.write_text('{"split":"train"}\n')
    readout = run / 'HOLDOUT_READOUT.json'
    readout.write_text(json.dumps({'checkpoint_sha256': checkpoint_sha}))
    gate = {
        'status': 'success', 'job_key': key, 'manifest_sha256': digest,
        'test_accessed': False,
        'readout_sha256': hashlib.sha256(readout.read_bytes()).hexdigest(),
        'checkpoint_sha256': checkpoint_sha,
        'data_flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest(),
        'selection_audit_sha256': None, 'replay_steps': 0,
    }
    (run / 'JOB_SUCCESS.json').write_text(json.dumps(gate))
    return hashlib.sha256((run / 'JOB_SUCCESS.json').read_bytes()).hexdigest()


class BalancedAdmissionTests(unittest.TestCase):
    def test_sealed_open_start_is_reconstructable_and_only_unstarted_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keys = ('cifar100/sample_only_off/none/70',
                    'cifar100/sample_only_off/none/71',
                    'cifar100/sample_only_off/none/72')
            rows = [{'job_key': key, 'config_sha256': str(i) * 64,
                     'method': 'sample_only_off'} for i, key in enumerate(keys, 1)]
            seal0 = sealed_job(root, keys[0], 'm')
            sealed_job(root, keys[1], 'm')
            events = [
                {'kind': 'start', 'job_key': keys[0], 'attempt': 1,
                 'config_sha256': rows[0]['config_sha256']},
                {'kind': 'finish', 'job_key': keys[0], 'attempt': 1,
                 'classification': 'success', 'job_success_sha256': seal0},
                {'kind': 'start', 'job_key': keys[1], 'attempt': 1,
                 'config_sha256': rows[1]['config_sha256']},
            ]
            state = balanced.validate_history(rows, events, root, 'm')
            self.assertEqual(state['pending'], [keys[2]])
            self.assertEqual(state['reconstructable'], [keys[1]])
            events[2]['config_sha256'] = 'x' * 64
            with self.assertRaises(ValueError):
                balanced.validate_history(rows, events, root, 'm')

    def test_training_anomaly_is_accounted_for_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = 'cifar100/sample_only_off/none/70'
            rows = [{'job_key': key, 'config_sha256': 'a' * 64,
                     'method': 'sample_only_off'}]
            events = [
                {'kind': 'start', 'job_key': key, 'attempt': 1,
                 'config_sha256': 'a' * 64},
                {'kind': 'finish', 'job_key': key, 'attempt': 1,
                 'classification': 'training_anomaly'},
            ]
            state = balanced.validate_history(rows, events, Path(tmp), 'm')
            self.assertEqual(state['pending'], [])
            self.assertEqual(state['anomalies'], [key])


if __name__ == '__main__':
    unittest.main()
