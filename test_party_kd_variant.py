import unittest
import json
import os
import subprocess
import sys
import tempfile

from config import validate_party_kd_variant


class PartyKDVariantTest(unittest.TestCase):
    def test_uniform_rejects_disabled_kd(self):
        cfg = {
            'dep_tracking_enabled': 0,
            'party_kd_enabled': 0,
            'party_kd_mode': 'uniform',
        }
        with self.assertRaisesRegex(ValueError, 'uniform'):
            validate_party_kd_variant(cfg, 'uniform')

    def test_uniform_accepts_effective_uniform_config(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'uniform',
        }
        validate_party_kd_variant(cfg, 'uniform')

    def test_static_rejects_uniform_mode(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'uniform',
        }
        with self.assertRaisesRegex(ValueError, 'static'):
            validate_party_kd_variant(cfg, 'static')

    def test_inverse_accepts_effective_inverse_config(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'inverse',
        }
        validate_party_kd_variant(cfg, 'inverse')

    def test_inverse_rejects_disabled_kd(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 0,
            'party_kd_mode': 'inverse',
        }
        with self.assertRaisesRegex(ValueError, 'inverse'):
            validate_party_kd_variant(cfg, 'inverse')

    def test_shuffled_requires_manifest_and_permutation_seed(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'shuffled',
            'party_weight_manifest': '',
            'party_shuffle_seed': -1,
        }
        with self.assertRaisesRegex(ValueError, 'manifest'):
            validate_party_kd_variant(cfg, 'shuffled')

    def test_shuffled_accepts_complete_contract(self):
        cfg = {
            'dep_tracking_enabled': 1,
            'party_kd_enabled': 1,
            'party_kd_mode': 'shuffled',
            'party_weight_manifest': 'weights.pt',
            'party_shuffle_seed': 101,
        }
        validate_party_kd_variant(cfg, 'shuffled')

    def test_audit_rejects_mislabeled_run(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = os.path.join(root, 'uniform_seed44')
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, 'config.json'), 'w') as f:
                json.dump({
                    'seed': 44,
                    'dep_tracking_enabled': 0,
                    'party_kd_enabled': 0,
                    'party_kd_mode': 'uniform',
                }, f)
            with open(os.path.join(run_dir, 'results.json'), 'w') as f:
                json.dump({'cl_metrics': {
                    'AA_final': 0.1,
                    'AA_cil': 0.2,
                    'BWT': -0.1,
                    'AA_final_taskil': 0.3,
                    'AA_cil_taskil': 0.4,
                }}, f)

            proc = subprocess.run(
                [sys.executable, 'audit_stage2_runs.py',
                 '--results_root', root,
                 '--run_glob', '*/config.json',
                 '--expected_party_kd_variant', 'uniform'],
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 1)
        self.assertIn('FAIL_VARIANT_CONTRACT', proc.stdout)

    def test_audit_does_not_warn_about_missing_optional_logs(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in (42, 43):
                run_dir = os.path.join(root, f'uniform_seed{seed}')
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, 'config.json'), 'w') as f:
                    json.dump({
                        'seed': seed,
                        'dep_tracking_enabled': 1,
                        'party_kd_enabled': 1,
                        'party_kd_mode': 'uniform',
                    }, f)
                with open(os.path.join(run_dir, 'results.json'), 'w') as f:
                    json.dump({'cl_metrics': {
                        'AA_final': 0.1,
                        'AA_cil': 0.2,
                        'BWT': -0.1,
                        'AA_final_taskil': 0.3,
                        'AA_cil_taskil': 0.4,
                    }}, f)

            proc = subprocess.run(
                [sys.executable, 'audit_stage2_runs.py',
                 '--results_root', root,
                 '--run_glob', '*/config.json',
                 '--expected_party_kd_variant', 'uniform'],
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertNotIn('duplicate run.log hashes', proc.stdout)

    def test_audit_accepts_inverse_run(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = os.path.join(root, 'inverse_seed42')
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, 'config.json'), 'w') as f:
                json.dump({
                    'seed': 42,
                    'dep_tracking_enabled': 1,
                    'party_kd_enabled': 1,
                    'party_kd_mode': 'inverse',
                }, f)
            with open(os.path.join(run_dir, 'results.json'), 'w') as f:
                json.dump({'cl_metrics': {
                    'AA_final': 0.1,
                    'AA_cil': 0.2,
                    'BWT': -0.1,
                    'AA_final_taskil': 0.3,
                    'AA_cil_taskil': 0.4,
                }}, f)

            proc = subprocess.run(
                [sys.executable, 'audit_stage2_runs.py',
                 '--results_root', root,
                 '--run_glob', '*/config.json',
                 '--expected_party_kd_variant', 'inverse'],
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == '__main__':
    unittest.main()
