"""Guards for training-only four-party resource-lite development trials."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from calibration_split import manifest_indices
from data_utils import VFLDataset
from run_resource_lite_pilot import (
    PilotDataset, StopBeforeTest, build_holdout_manifest, derive_config,
    expected_tasks,
    validate_source_data_hash, validate_training_only_artifacts,
)


class ResourceLitePilotTests(unittest.TestCase):
    def source(self, dataset='isolet'):
        paths = {
            'isolet': ('isolet/isolet_vfl.npz', 26, 13, 0),
            'upmc_food101': (
                'upmc_food101_image_clean_4party_v1/upmc_food101_vfl.npz',
                101, 10, 0,
            ),
            'cifar100': ('', 100, 10, 1),
        }
        vector, classes, tasks, bic = paths[dataset]
        return {
            'seed': 42, 'data': 'cifar100' if bic else 'tabvfl',
            'num_classes': classes, 'num_tasks': tasks, 'num_parties': 4,
            'vector_npz': '/brf/users/yangxx/VF-CL/data/' + vector if vector else '',
            'data_path': '/brf/users/yangxx/VF-CL/data',
            'cl_method': 'proto_evolve', 'sample_replay_only': 1,
            'head_consolidation_enabled': 1,
            'head_consolidation_schedule': 'final',
            'head_consolidation_mode': 'adaptive_dual_branch',
            'head_consolidation_samples_per_class': 20,
            'head_consolidation_steps': 500,
            'custom_tasks': '' if bic else expected_tasks(dataset),
            'classes_per_task': 10 if bic else 2,
            'epochs_per_task': 50,
            'formal_deferred_evaluation': True,
            'proto_replay_gradient_enabled': 0, 'proto_sdc': False,
            'feat_distill_weight': 0.0,
            'distill_weight': 0.25, 'party_kd_enabled': 1,
            'party_kd_mode': 'uniform',
            'dep_tracking_enabled': 1, 'expected_party_kd_variant': 'uniform',
            'lambda_validation_enabled': 1, 'bic_enabled': bic,
            'results_dir': '/old', 'output_dir': '/old/run',
            'exp_name': 'old',
        }

    def test_pair_only_changes_declared_resource_options_and_paths(self):
        source = self.source()
        with tempfile.TemporaryDirectory() as folder:
            reference = derive_config(source, folder, 'isolet', 54, 'reference20')
            lite = derive_config(source, folder, 'isolet', 54, 'lite18')
        self.assertEqual(reference['head_consolidation_samples_per_class'], 20)
        self.assertEqual(lite['head_consolidation_samples_per_class'], 18)
        self.assertEqual(lite['head_consolidation_mode'], 'full_classifier')
        self.assertEqual(lite['resource_lite'], 1)
        self.assertEqual(lite['distill_weight'], 0.0)
        self.assertEqual(lite['party_kd_enabled'], 0)
        self.assertEqual(lite['dep_tracking_enabled'], 0)
        self.assertEqual(lite['lambda_validation_enabled'], 0)
        self.assertEqual(reference['lambda_validation_enabled'], 1)
        self.assertEqual(source['seed'], 42)
        self.assertEqual(source['head_consolidation_mode'], 'adaptive_dual_branch')

    def test_rejects_two_party_or_wrong_upmc_cache(self):
        source = self.source('upmc_food101')
        with tempfile.TemporaryDirectory() as folder:
            for change in (
                    {'num_parties': 2},
                    {'vector_npz': '/old/upmc_food101_vfl.npz'},
                    {'num_classes': 100},
            ):
                with self.subTest(change=change):
                    with self.assertRaisesRegex(ValueError, 'dataset identity'):
                        derive_config({**source, **change}, folder,
                                      'upmc_food101', 54, 'lite18')

    def test_rejects_config_drift_and_wrong_seed(self):
        source = self.source()
        with tempfile.TemporaryDirectory() as folder:
            for change in ({'head_consolidation_steps': 600},
                           {'sample_replay_only': 0},
                           {'formal_deferred_evaluation': False}):
                with self.subTest(change=change):
                    with self.assertRaises(ValueError):
                        derive_config({**source, **change}, folder,
                                      'isolet', 54, 'lite18')
            with self.assertRaisesRegex(ValueError, 'seed'):
                derive_config(source, folder, 'isolet', 42, 'lite18')

    def test_smoke_uses_first_two_tasks_and_one_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            config = derive_config(
                self.source(), folder, 'isolet', 54, 'lite18', smoke=True)
        self.assertTrue(config['pilot_smoke'])
        self.assertEqual(config['num_tasks'], 2)
        self.assertEqual(config['custom_tasks'], '0,1|2,3')
        self.assertEqual(config['epochs_per_task'], 1)
        self.assertTrue(config['output_dir'].endswith('_smoke'))

    def test_holdout_excludes_calibration_and_validation(self):
        labels = [0] * 100 + [1] * 100
        excluded = set(range(10)) | set(range(100, 110))
        manifest, indices = build_holdout_manifest(
            labels, excluded, 10, 20261013, 'toy')
        self.assertEqual(indices, manifest_indices(manifest))
        self.assertEqual(len(indices), 20)
        self.assertFalse(indices & excluded)

    def test_test_loader_and_authorization_are_blocked(self):
        dataset = PilotDataset.__new__(PilotDataset)
        dataset._formal_durable_state = lambda: ([], None)
        with mock.patch.object(dataset, '_validate_formal_phase_transition'), \
                mock.patch.object(VFLDataset, 'authorize_formal_access',
                                  side_effect=AssertionError('test accessed')):
            with self.assertRaises(StopBeforeTest):
                dataset.authorize_formal_access(split='test')
        with self.assertRaisesRegex(AssertionError, 'test loader'):
            dataset.get_test_loader([0, 1])

    def test_train_loader_removes_holdout_indices(self):
        dataset = PilotDataset.__new__(PilotDataset)
        dataset.holdout_indices = {2, 4}
        with mock.patch.object(VFLDataset, '_loader', return_value='loader') as base:
            self.assertEqual(dataset._loader(object(), [1, 2, 3, 4],
                                             ('train', (0,)), True), 'loader')
            self.assertEqual(base.call_args.args[1], [1, 3])

    def test_vector_hash_must_match_frozen_data(self):
        with tempfile.TemporaryDirectory() as folder:
            vector = Path(folder) / 'isolet_vfl.npz'
            vector.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'hash'):
                validate_source_data_hash({
                    'pilot_dataset': 'isolet', 'vector_npz': str(vector),
                })

    def test_rejects_any_formal_test_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'checkpoints').mkdir()
            (run / 'checkpoints' / 'formal_final.pt').write_bytes(b'checkpoint')
            (run / 'FORMAL_STATE_FROZEN.json').write_text('{}')
            (run / 'FORMAL_EVALUATION_PENDING.json').write_text('{}')
            (run / 'data_flow_audit.jsonl').write_text('{"split":"train"}\n')
            validate_training_only_artifacts(run)
            (run / 'results.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'test access'):
                validate_training_only_artifacts(run)


if __name__ == '__main__':
    unittest.main()
