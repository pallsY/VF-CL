import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from diagnose_sampleonly_head import validate_run_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagnosticInputTests(unittest.TestCase):
    def test_rejects_changed_holdout_or_test_access(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            (run / 'pilot_holdout').mkdir()
            (run / 'checkpoints').mkdir()
            (run / 'FORMAL_EVALUATION_PENDING.json').write_text('{}')
            config = run / 'config.json'
            config.write_text(json.dumps({
                'pilot_dataset': 'isolet', 'pilot_variant': 'reference20',
                'seed': 54, 'num_tasks': 13,
            }))
            manifest = run / 'pilot_holdout' / 'manifest.json'
            manifest.write_text('{}')
            final = run / 'checkpoints' / 'formal_final.pt'
            final.write_bytes(b'final')
            (run / 'checkpoints' / 'event_12_CIL.pt').write_bytes(b'pre')
            (run / 'HOLDOUT_READOUT.json').write_text(json.dumps({
                'status': 'independent_training_holdout_readout',
                'dataset': 'isolet', 'variant': 'reference20', 'seed': 54,
                'config_sha256': digest(config),
                'holdout_manifest_sha256': digest(manifest),
                'checkpoint_sha256': digest(final),
                'metrics': {'cil': 0.9},
            }))
            self.assertEqual(validate_run_inputs(run)[2].name,
                             'event_12_CIL.pt')
            manifest.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'holdout manifest hash'):
                validate_run_inputs(run)
            manifest.write_text('{}')
            final.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'final checkpoint hash'):
                validate_run_inputs(run)
            final.write_bytes(b'final')
            (run / 'results.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'test access'):
                validate_run_inputs(run)


if __name__ == '__main__':
    unittest.main()
