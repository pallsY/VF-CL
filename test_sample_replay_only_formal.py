"""Contract checks for the separately audited raw-sample-memory ablation."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import config
import cl_methods.proto_evolve as proto_evolve
import adaptive9_matrix as matrix
import replay11_matrix as baseline
import three_dataset_formal_registry as registry
from pathlib import Path
from test_proto_evolve_adaptive_head import TinyTrainer


def sample_args(output_dir):
    return SimpleNamespace(
        cl_method='proto_evolve', num_parties=2, num_classes=2,
        num_tasks=2, seed=42, device='cpu', batch_size=2,
        data='synthvfl', party_col_ranges=[(0, 2), (2, 4)],
        aggregation='sum', output_dir=output_dir,
        head_consolidation_enabled=1,
        head_consolidation_schedule='final',
        head_consolidation_mode='adaptive_dual_branch',
        head_consolidation_samples_per_class=20,
        unlearn_after_tasks=[], unlearn_classes=[],
        sample_replay_only=1, proto_replay_gradient_enabled=0,
        proto_sdc=False, feat_distill_weight=0.0,
        distill_weight=0.25,
    )


class SampleReplayOnlyFormalTest(unittest.TestCase):
    def test_nine_commands_match_formal_adaptive_except_declared_switches(self):
        switches = {
            'sample_replay_only': '1',
            'proto_replay_gradient_enabled': '0',
            'proto_sdc': '0',
            'feat_distill_weight': '0.0',
        }
        for dataset in baseline.DATASETS:
            for seed in baseline.SEEDS:
                with self.subTest(dataset=dataset, seed=seed):
                    command = matrix.command_for(
                        dataset, seed, python=Path('/python'),
                        worktree=Path('/code'), results_dir=Path('/results'),
                        data_root=Path('/data'), device='cuda:0')
                    actual = dict(zip(
                        (token[2:] for token in command[3::2]),
                        command[4::2],
                    ))
                    expected = dict(registry.protocol_for(
                        registry.FormalSpec(dataset, 'adaptive', seed)
                    )['base_options'])
                    expected.update(
                        data_path=str(Path('/data')),
                        results_dir=str(Path('/results')),
                        device='cuda:0', exp_name='run',
                        formal_deferred_evaluation=True,
                    )
                    if dataset == 'upmc_food101':
                        expected['num_parties'] = 4
                        expected['vector_npz'] = str(
                            Path('/data') / baseline.UPMC_FOUR_CACHE)
                    elif dataset == 'isolet':
                        expected['vector_npz'] = str(
                            Path('/data') / 'isolet/isolet_vfl.npz')
                    expected = {
                        key: registry._option_token(key, value)
                        for key, value in expected.items()
                    }
                    for key in switches:
                        expected.pop(key, None)
                    self.assertEqual({k: v for k, v in actual.items()
                                      if k not in switches}, expected)
                    self.assertEqual({k: actual.get(k) for k in switches},
                                     switches)
                    self.assertEqual(actual['distill_weight'], '0.25')
                    self.assertEqual(actual['num_parties'], '4')

    def test_validator_rejects_any_feature_replay_path(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = sample_args(output_dir)
            config.validate_sample_replay_only(args)
            self.assertEqual(args.distill_weight, 0.25)
            incompatible = {
                'cl_method': 'er',
                'head_consolidation_enabled': 0,
                'head_consolidation_mode': 'full_classifier',
                'head_consolidation_schedule': 'every',
                'head_consolidation_samples_per_class': 40,
                'proto_replay_gradient_enabled': 1,
                'proto_sdc': True,
                'feat_distill_weight': 0.05,
            }
            for field, value in incompatible.items():
                with self.subTest(field=field):
                    changed = sample_args(output_dir)
                    setattr(changed, field, value)
                    with self.assertRaisesRegex(ValueError, field):
                        config.validate_sample_replay_only(changed)

    def test_final_head_uses_raw_classes_and_no_prototypes(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = proto_evolve.ProtoEvolveCL(
                TinyTrainer(), sample_args(output_dir))
            method.head_raw_replay = {
                0: torch.tensor([[1., 2., 3., 4.]]),
                1: torch.tensor([[2., 1., 4., 3.]]),
            }
            method.head_task_classes = {0: [0], 1: [1]}
            self.assertEqual(method.global_protos, {})
            with mock.patch.object(proto_evolve, 'fit_adaptive_candidates',
                                   side_effect=RuntimeError('captured')) as fit:
                with self.assertRaisesRegex(RuntimeError, 'captured'):
                    method._consolidate_head(1)
            self.assertEqual(fit.call_args.args[2], {})
            self.assertEqual(fit.call_args.args[0].classifier.out_features, 2)

    def test_rejects_checkpoint_with_prototype_memory(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = proto_evolve.ProtoEvolveCL(
                TinyTrainer(), sample_args(output_dir))
            with self.assertRaisesRegex(ValueError, 'prototypes'):
                method.load_state({'global_protos': {0: {'mean': 1}}})

    def test_task_boundary_keeps_only_bounded_raw_memory(self):
        with tempfile.TemporaryDirectory() as output_dir:
            method = proto_evolve.ProtoEvolveCL(
                TinyTrainer(), sample_args(output_dir))
            method.current_task_classes = [0]
            moments = {0: {'mean': torch.zeros(2), 'std': torch.ones(2)}}
            raw = {0: torch.zeros(20, 4)}
            with mock.patch.object(method, '_compute_fim'), \
                    mock.patch.object(method, '_compute_protos',
                                      return_value=(moments, raw)), \
                    mock.patch.object(method, '_consolidate_head'):
                method.after_task(None, 0)
            self.assertEqual(method.global_protos, {})
            self.assertEqual(method.prev_protos, {})
            self.assertEqual(method.head_raw_replay[0].shape[0], 20)


if __name__ == '__main__':
    unittest.main()
