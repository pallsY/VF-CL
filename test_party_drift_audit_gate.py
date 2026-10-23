import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from determinism import derive_seed
from party_drift_local import replay_step_audit
from party_drift_replay import BalancedOldReplay


class SelectionAuditGateTests(unittest.TestCase):
    def test_cmcdr_fields_are_mandatory_for_direct_control(self):
        from party_drift_audit_gate import validate_cmcdr_fields
        with self.assertRaises(ValueError):
            validate_cmcdr_fields({'replay_arm': 'cmcdr'}, 'cmcdr')

    def test_probe_ids_are_replayed_from_locked_seed(self):
        from party_drift_audit_gate import validate_probe_ids
        import numpy as np
        rng = np.random.default_rng(np.random.SeedSequence([70, 1, 0, 90210]))
        expected = [[class_id, int(row)] for class_id in (0, 1)
                    for row in rng.choice(20, 2, replace=False)]
        probe = {'logical_probe_ids': expected,
                 'seed_sequence': [70, 1, 0, 90210]}
        validate_probe_ids(probe, 70, 1, 0, [0, 1])
        probe['logical_probe_ids'][0][1] = (expected[0][1] + 1) % 20
        with self.assertRaises(ValueError):
            validate_probe_ids(probe, 70, 1, 0, [0, 1])

    def test_sealed_job_rejects_missing_or_changed_selection_ledger(self):
        from party_drift_audit_gate import verify_sealed_job
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            checkpoint = run / 'checkpoints' / 'formal_final.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'checkpoint')
            flow = run / 'data_flow_audit.jsonl'
            flow.write_text('{"split":"train"}\n')
            import hashlib
            readout = run / 'HOLDOUT_READOUT.json'
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            readout.write_text(json.dumps({'checkpoint_sha256': checkpoint_sha}))
            gate = {'status': 'success', 'job_key': 'isolet/drift/0.5/70',
                    'manifest_sha256': 'm', 'test_accessed': False,
                    'readout_sha256': hashlib.sha256(readout.read_bytes()).hexdigest(),
                    'checkpoint_sha256': checkpoint_sha,
                    'data_flow_sha256': hashlib.sha256(flow.read_bytes()).hexdigest(),
                    'selection_audit_sha256': None, 'replay_steps': 2}
            (run / 'JOB_SUCCESS.json').write_text(json.dumps(gate))
            with self.assertRaises(ValueError):
                verify_sealed_job(run, gate['job_key'], 'm', replay=True)
            audit = run / 'party_drift_audit' / 'selection.jsonl'
            audit.parent.mkdir()
            audit.write_text('{"selected":[0,1]}\n')
            gate['selection_audit_sha256'] = hashlib.sha256(audit.read_bytes()).hexdigest()
            (run / 'JOB_SUCCESS.json').write_text(json.dumps(gate))
            verify_sealed_job(run, gate['job_key'], 'm', replay=True)
            audit.write_text('{"selected":[0,2]}\n')
            with self.assertRaises(ValueError):
                verify_sealed_job(run, gate['job_key'], 'm', replay=True)
            audit.write_text('{"selected":[0,1]}\n')
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                verify_sealed_job(run, gate['job_key'], 'm', replay=True)

    def test_exact_step_and_sample_audit_required_before_success(self):
        from party_drift_audit_gate import verify_selection_audit

        args = SimpleNamespace(party_drift_enabled=1, party_replay_arm='uniform',
                               party_replay_alpha=0.5, seed=70, num_tasks=2,
                               epochs_per_task=1, batch_size=2,
                               num_parties=4, embed_dim=2)
        dataset = SimpleNamespace(
            trainset=SimpleNamespace(targets=[0, 0, 1, 1, 2, 2, 3, 3]),
            holdout_indices=set(), pilot_pair_excluded=set())
        classes = {0: [0, 1], 1: [2, 3]}
        sampler = BalancedOldReplay(
            {class_id: torch.arange(20) for class_id in (0, 1)},
            derive_seed(70, 'party_drift_sampler', 1))
        rows = [{'kind': 'weight_refresh', 'task': 1, 'epoch': 0,
                 'arm': 'uniform', 'weights': [0.25] * 4}]
        for step in range(2):
            _, _, identities, duplicates = sampler.sample()
            record = replay_step_audit([0.25] * 4, 2, 70, 1, 0, step, 4,
                                       embedding_dims=[2] * 4, replay_batch=64)
            record.update(logical_replay_ids=identities,
                          duplicate_samples=duplicates, replay_arm='uniform',
                          replay_alpha=0.5)
            rows.append(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'selection.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(verify_selection_audit(
                path, args, dataset, classes)['replay_steps'], 2)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
            with self.assertRaises(ValueError):
                verify_selection_audit(path, args, dataset, classes)


if __name__ == '__main__':
    unittest.main()
