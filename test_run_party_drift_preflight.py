import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_party_drift_preflight import derive_config


class PartyDriftPreflightConfigTest(unittest.TestCase):
    def test_derives_only_two_training_tasks_without_test_access(self):
        from run_resource_lite_pilot import expected_tasks
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.json'
            source.write_text(json.dumps({
                'data': 'tabvfl', 'num_classes': 26, 'num_tasks': 13,
                'custom_tasks': expected_tasks('isolet'),
                'num_parties': 4, 'seed': 42,
                'vector_npz': '/brf/users/yangxx/VF-CL/data/isolet/isolet_vfl.npz',
                'sample_replay_only': 1,
                'head_consolidation_samples_per_class': 20,
                'formal_deferred_evaluation': True,
            }), encoding='utf-8')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch('run_party_drift_preflight.SOURCE_CONFIG_SHA256',
                       {'isolet': digest}):
                config = derive_config(source, Path(tmp) / 'new', 'isolet')
            self.assertEqual(config['seed'], 73)
            self.assertEqual(config['num_tasks'], 2)
            self.assertEqual(config['custom_tasks'], '0,1|2,3')
            self.assertEqual(config['epochs_per_task'], 1)
            self.assertEqual(config['device'], 'cpu')
            self.assertEqual(config['pilot_holdout_per_class'], 30)
            self.assertEqual(config['pilot_holdout_seed'], 20261023)
            self.assertEqual(config['party_drift_enabled'], 1)
            self.assertEqual(config['party_drift_preflight'], 1)


if __name__ == '__main__':
    unittest.main()
