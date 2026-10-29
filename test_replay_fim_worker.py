import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

import run_replay_fim_development as worker


class ReplayFIMWorkerTests(unittest.TestCase):
    def test_sources_are_byte_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / name for name in ('source', 'data', 'holdout', 'validation')}
            for name, path in paths.items():
                path.write_text(name, encoding='utf-8')
            config = {
                'pilot_protocol': 'allparty-fim-final-attempt-v1',
                'development_job_key': 'isolet/f1/0.25/80',
                'development_source_config_path': str(paths['source']),
                'development_source_data_path': str(paths['data']),
                'development_holdout_manifest': str(paths['holdout']),
                'pilot_paired_validation_manifest': str(paths['validation']),
            }
            row = {'job_key': config['development_job_key'], **{
                field: hashlib.sha256(paths[name].read_bytes()).hexdigest()
                for field, name in (
                    ('source_config_sha256', 'source'),
                    ('source_data_sha256', 'data'),
                    ('holdout_sha256', 'holdout'),
                    ('validation_manifest_sha256', 'validation'))}}
            worker.validate_job_sources(config, row)
            paths['data'].write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                worker.validate_job_sources(config, row)

    def test_fim_mask_audit_rejects_off_freeze_and_cumulative_unfreeze(self):
        states = [
            {'task_id': 0, 'trainer_state': {'bottoms': [
                {'weight': torch.zeros(4), 'bias': torch.zeros(1)}]},
             'cl_state': {'fim_masks': [{'weight': True, 'bias': False}]}},
            {'task_id': 1, 'trainer_state': {'bottoms': [
                {'weight': torch.zeros(4), 'bias': torch.zeros(1)}]},
             'cl_state': {'fim_masks': [{'weight': False, 'bias': True}]}},
        ]
        fresh = worker.audit_fim_masks(states, 'fresh')
        self.assertEqual(fresh[0]['parties'][0]['frozen_names'], ['weight'])
        self.assertEqual(fresh[1]['parties'][0]['frozen_param_fraction'], 0.2)
        with self.assertRaises(ValueError):
            worker.audit_fim_masks(states, 'cumulative')
        with self.assertRaises(ValueError):
            worker.audit_fim_masks(states, 'off')
        off = [{**state, 'cl_state': {'fim_masks': [
            {'weight': False, 'bias': False}]}} for state in states]
        self.assertEqual(len(worker.audit_fim_masks(off, 'off')), 2)


if __name__ == '__main__':
    unittest.main()
