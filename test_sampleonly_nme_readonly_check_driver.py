"""run() must only ever read already-frozen training-holdout artifacts and
must never write into the source run directory."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from run_sampleonly_nme_readonly_check import run


class NmeReadonlyDriverTests(unittest.TestCase):
    def _frozen_run(self, tmp, status='independent_training_holdout_readout',
                    samples_per_class=20):
        run_dir = Path(tmp) / 'source' / 'isolet_seed54_reference20'
        run_dir.mkdir(parents=True)
        (run_dir / 'FORMAL_STATE_FROZEN.json').write_text('{}')
        (run_dir / 'FORMAL_EVALUATION_PENDING.json').write_text('{}')
        (run_dir / 'checkpoints').mkdir()
        (run_dir / 'checkpoints' / 'formal_final.pt').write_bytes(b'stub')
        (run_dir / 'data_flow_audit.jsonl').write_text(
            '{"split": "train"}\n')
        (run_dir / 'pilot_holdout').mkdir()
        (run_dir / 'pilot_holdout' / 'manifest.json').write_bytes(b'manifest')
        config = {
            'output_dir': str(run_dir), 'pilot_dataset': 'isolet',
            'seed': 54, 'pilot_variant': 'reference20',
            'head_consolidation_samples_per_class': samples_per_class,
        }
        (run_dir / 'config.json').write_text(json.dumps(config))
        from run_resource_lite_pilot import sha256
        (run_dir / 'HOLDOUT_READOUT.json').write_text(json.dumps({
            'status': status,
            'dataset': 'isolet', 'seed': 54, 'variant': 'reference20',
            'config_sha256': sha256(run_dir / 'config.json'),
            'checkpoint_sha256': sha256(
                run_dir / 'checkpoints' / 'formal_final.pt'),
            'freeze_sha256': sha256(run_dir / 'FORMAL_STATE_FROZEN.json'),
            'holdout_manifest_sha256': sha256(
                run_dir / 'pilot_holdout' / 'manifest.json'),
            'metrics': {'cil': 0.9},
        }))
        return run_dir

    def _diag_root(self, tmp):
        return Path(tmp) / 'diagnostic'

    def _assert_preflight_rejects(self, run_dir, diagnostic_root, pattern):
        try:
            run(run_dir, diagnostic_root)
        except ValueError as error:
            self.assertRegex(str(error), pattern)
        except Exception as error:
            self.fail(f'preflight reached data loading: {error}')
        else:
            self.fail('preflight accepted a drifted source')

    def test_rejects_run_still_open_for_formal_test(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'open'
            run_dir.mkdir()
            (run_dir / 'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                run(run_dir, self._diag_root(tmp))

    def test_rejects_run_that_already_reached_test_access(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            (run_dir / 'results.json').write_text('{}')
            with self.assertRaises(ValueError):
                run(run_dir, self._diag_root(tmp))

    def test_rejects_config_output_dir_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            config = json.loads((run_dir / 'config.json').read_text())
            config['output_dir'] = str(run_dir) + '_other'
            (run_dir / 'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(run_dir, self._diag_root(tmp))

    def test_rejects_non_reference20_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp, samples_per_class=18)
            self._assert_preflight_rejects(
                run_dir, self._diag_root(tmp), 'reference20')

    def test_rejects_wrong_variant_even_with_twenty_samples(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            config_path = run_dir / 'config.json'
            config = json.loads(config_path.read_text())
            config['pilot_variant'] = 'lite18'
            config_path.write_text(json.dumps(config))
            self._assert_preflight_rejects(
                run_dir, self._diag_root(tmp), 'reference20')

    def test_rejects_checkpoint_that_drifted_before_scoring(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            (run_dir / 'checkpoints' / 'formal_final.pt').write_bytes(b'drifted')
            self._assert_preflight_rejects(
                run_dir, self._diag_root(tmp), 'checkpoint.*frozen readout')

    def test_rejects_config_that_drifted_before_scoring(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            config_path = run_dir / 'config.json'
            config = json.loads(config_path.read_text())
            config['extra'] = 'drifted'
            config_path.write_text(json.dumps(config))
            self._assert_preflight_rejects(
                run_dir, self._diag_root(tmp), 'config.*frozen readout')

    def test_rejects_freeze_record_that_drifted_before_scoring(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            (run_dir / 'FORMAL_STATE_FROZEN.json').write_text('{"drift": true}')
            self._assert_preflight_rejects(
                run_dir, self._diag_root(tmp), 'freeze.*frozen readout')

    def test_rejects_non_frozen_readout_status(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp, status='full_schedule_one_epoch_smoke')
            with self.assertRaisesRegex(ValueError, 'frozen training-holdout'):
                run(run_dir, self._diag_root(tmp))

    def test_rejects_diagnostic_root_inside_source_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            with self.assertRaisesRegex(ValueError, 'outside'):
                run(run_dir, run_dir / 'nested')

    def test_rejects_diagnostic_root_equal_to_source_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            with self.assertRaisesRegex(ValueError, 'outside'):
                run(run_dir, run_dir)

    def test_source_run_is_never_written_to(self):
        """Every file under the source run directory must be byte-identical
        before and after a (failing, mocked) scoring attempt."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            before = {
                path: path.read_bytes()
                for path in run_dir.rglob('*') if path.is_file()
            }
            class FaithfulPilotDataset:
                def __init__(self, args):
                    (Path(args.output_dir) / 'pilot_holdout').mkdir(
                        parents=True, exist_ok=True)
                    (Path(args.output_dir) / 'pilot_holdout' / 'manifest.json'
                    ).write_bytes((run_dir / 'pilot_holdout' / 'manifest.json'
                                  ).read_bytes())

            with patch('run_sampleonly_nme_readonly_check.PilotDataset',
                       FaithfulPilotDataset), \
                 patch('run_sampleonly_nme_readonly_check.nme_readout',
                       side_effect=RuntimeError('stop before scoring')):
                with self.assertRaises(RuntimeError):
                    run(run_dir, self._diag_root(tmp))
            after = {
                path: path.read_bytes()
                for path in run_dir.rglob('*') if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(set(before), set(after))

    def test_does_not_write_holdout_manifest_into_source_run(self):
        """PilotDataset must be pointed at the diagnostic root, not the
        source run's output_dir, or it would overwrite the frozen manifest
        in place (the exact bug this test guards against)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            seen_output_dirs = []

            class RecordingPilotDataset:
                def __init__(self, args):
                    seen_output_dirs.append(args.output_dir)
                    (Path(args.output_dir) / 'pilot_holdout').mkdir(
                        parents=True, exist_ok=True)
                    (Path(args.output_dir) / 'pilot_holdout' / 'manifest.json'
                    ).write_bytes((run_dir / 'pilot_holdout' / 'manifest.json'
                                  ).read_bytes())

                def holdout_loader(self, classes):
                    return []

            with patch('run_sampleonly_nme_readonly_check.PilotDataset',
                       RecordingPilotDataset), \
                 patch('run_sampleonly_nme_readonly_check.nme_readout',
                       side_effect=RuntimeError('stop before scoring')):
                with self.assertRaises(RuntimeError):
                    run(run_dir, self._diag_root(tmp))
            self.assertEqual(len(seen_output_dirs), 1)
            self.assertNotEqual(Path(seen_output_dirs[0]).resolve(),
                               run_dir.resolve())
            self.assertEqual(Path(seen_output_dirs[0]).resolve(),
                             self._diag_root(tmp).resolve())

    def test_rejects_reconstructed_holdout_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)

            class WrongManifestPilotDataset:
                def __init__(self, args):
                    (Path(args.output_dir) / 'pilot_holdout').mkdir(
                        parents=True, exist_ok=True)
                    (Path(args.output_dir) / 'pilot_holdout' / 'manifest.json'
                    ).write_bytes(b'not the frozen manifest')

            with patch('run_sampleonly_nme_readonly_check.PilotDataset',
                       WrongManifestPilotDataset):
                with self.assertRaisesRegex(ValueError, 'reconstructed holdout differs'):
                    run(run_dir, self._diag_root(tmp))

    def test_rejects_run_whose_manifest_already_drifted_from_its_readout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._frozen_run(tmp)
            (run_dir / 'pilot_holdout' / 'manifest.json').write_bytes(
                b'drifted after the readout was written')
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                run(run_dir, self._diag_root(tmp))


if __name__ == '__main__':
    unittest.main()
