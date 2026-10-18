"""Frozen resource cells reject changed authorities and avoid test metrics."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import resource_ledger


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ResourceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.job = self.base / 'full/isolet/er/42'
        self.run = self.job / 'run'
        (self.run / 'formal_snapshots').mkdir(parents=True)
        (self.run / 'checkpoints').mkdir()
        audit = self.base / 'audit'
        audit.mkdir()
        (audit / 'attestations.jsonl').write_bytes(b'fixed-ledger\n')
        self.gate = {
            'producer_commit': 'a' * 40, 'audit_commit': 'b' * 40,
            'basic_passed': 99, 'numeric_passed': 33,
            'ledger_sha256': digest(b'fixed-ledger\n'),
        }
        (audit / 'FINAL_GATE.json').write_text(json.dumps(self.gate))
        self.authority = {
            'name': 'baseline', 'base': self.base,
            'jobs_root': self.base / 'full',
            'gate_path': audit / 'FINAL_GATE.json',
            'gate_sha256': digest((audit / 'FINAL_GATE.json').read_bytes()),
            'producer_commit': 'a' * 40, 'audit_commit': 'b' * 40,
            'basic_passed': 99, 'numeric_passed': 33,
            'methods': ('er',), 'data_root': '/data',
        }
        self.config = {
            'cl_method': 'er', 'seed': 42, 'num_tasks': 13,
            'num_parties': 4, 'epochs_per_task': 50,
            'data_path': '/data',
        }
        (self.run / 'config.json').write_text(json.dumps(self.config))
        (self.job / 'JOB_PLAN.json').write_text(json.dumps({
            'code_commit': 'a' * 40, 'dataset': 'isolet',
            'method': 'er', 'seed': 42,
        }))
        (self.run / 'checkpoints/formal_final.pt').write_bytes(b'final')
        (self.job / 'JOB_SUCCESS.json').write_text(json.dumps({
            'code_commit': 'a' * 40,
            'checkpoint_sha256': digest(b'final'),
            'metrics': {'aa_final_cil': 0.123456789},
        }))
        snapshots = []
        for task in range(13):
            name = f'formal_snapshots/event_{task}_CIL.pt'
            content = f'snapshot-{task}'.encode()
            (self.run / name).write_bytes(content)
            snapshots.append({'file': {'path': name,
                                       'sha256': digest(content)}})
        (self.run / 'FORMAL_STATE_FROZEN.json').write_text(json.dumps({
            'identity': {'snapshots': snapshots,
                         'protocol': {'source_provenance': {
                             'source_commit': 'a' * 40}}},
        }))

    @staticmethod
    def payload(path):
        name = Path(path).name
        task = 12 if name == 'formal_final.pt' else int(
            name.split('_')[1])
        return {
            'schema_version': 4, 'task_id': task,
            'trainer_state': {'bottoms': [torch.zeros(2) for _ in range(4)],
                              'top_model': torch.zeros(3)},
            'cl_state': {'data': {0: torch.zeros(2, 3)}},
        }

    def read(self):
        with mock.patch.object(resource_ledger, '_safe_torch_load',
                               side_effect=self.payload):
            return resource_ledger.read_cell(
                'isolet', 'er', 42, self.authority)

    def test_pinned_cell_counts_final_and_task_max_without_accuracy(self):
        row = self.read()
        self.assertEqual(row['job'], 'isolet/er/42')
        self.assertEqual(row['final_raw_examples'], 2)
        self.assertEqual(row['final_raw_bytes'], 24)
        self.assertEqual(row['final_model_bytes'], 44)
        self.assertEqual(row['max_learning_bytes'], 68)
        self.assertNotIn('metrics', row)
        self.assertNotIn('aa_final_cil', row)
        self.assertNotIn('0.123456789', json.dumps(row))

    def test_wrong_gate_or_producer_is_rejected(self):
        self.authority['gate_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'gate'):
            self.read()
        self.authority['gate_sha256'] = digest(
            (self.base / 'audit/FINAL_GATE.json').read_bytes())
        success_path = self.job / 'JOB_SUCCESS.json'
        success = json.loads(success_path.read_text())
        success['code_commit'] = 'c' * 40
        success_path.write_text(json.dumps(success))
        with self.assertRaisesRegex(ValueError, 'producer'):
            self.read()

    def test_legacy_gate_requires_separately_pinned_ledger_hash(self):
        gate_path = self.base / 'audit/FINAL_GATE.json'
        self.gate.pop('ledger_sha256')
        gate_path.write_text(json.dumps(self.gate))
        self.authority['gate_sha256'] = digest(gate_path.read_bytes())
        self.authority['ledger_sha256'] = digest(b'fixed-ledger\n')
        self.assertEqual(self.read()['final_raw_examples'], 2)
        self.authority['ledger_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'ledger'):
            self.read()

    def test_changed_checkpoint_and_wrong_party_count_are_rejected(self):
        (self.run / 'checkpoints/formal_final.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.read()
        (self.run / 'checkpoints/formal_final.pt').write_bytes(b'final')
        self.config['num_parties'] = 2
        (self.run / 'config.json').write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, 'four.party'):
            self.read()

    def test_changed_snapshot_and_missing_cell_are_rejected(self):
        (self.run / 'formal_snapshots/event_0_CIL.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            self.read()
        with self.assertRaisesRegex(ValueError, 'matrix'):
            with mock.patch.object(resource_ledger, '_safe_torch_load',
                                   side_effect=self.payload):
                resource_ledger.read_cell(
                    'isolet', 'er', 45, self.authority)


if __name__ == '__main__':
    unittest.main()
