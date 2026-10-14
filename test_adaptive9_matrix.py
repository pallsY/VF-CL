"""Fairness checks for the nine proposed-method comparison cells."""

import unittest
import tempfile
from pathlib import Path

import replay11_matrix as baseline
import three_dataset_formal_registry as registry

from adaptive9_matrix import command_for, jobs, run_one


def options(command):
    tokens = command[3:]
    return dict(zip((token[2:] for token in tokens[::2]), tokens[1::2]))


class Adaptive9CommandTests(unittest.TestCase):
    def test_existing_cell_cannot_be_relaunched(self):
        with tempfile.TemporaryDirectory() as scratch:
            (Path(scratch) / 'isolet/adaptive/42').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_one('isolet', 42, results_root=Path(scratch),
                        data_root=Path(scratch) / 'missing', device='cpu')

    def test_nine_cells_are_disjoint_from_baselines(self):
        cells = jobs()
        self.assertEqual(len(cells), 9)
        self.assertEqual(len(set(cells)), 9)
        self.assertTrue(all(method == 'adaptive'
                            for _, method, _ in cells))
        self.assertTrue(set(cells).isdisjoint(baseline.jobs()))

    def test_shared_protocol_matches_frozen_baselines(self):
        shared = (
            'num_classes', 'num_tasks', 'custom_tasks', 'num_parties',
            'model_type', 'aggregation', 'epochs_per_task', 'batch_size',
            'optimizer', 'lr', 'bottom_lr_scale', 'task_ce_mode', 'momentum',
            'weight_decay', 'bic_enabled', 'bic_fit_mode', 'bic_per_class',
            'bic_split_seed', 'bic_steps', 'lambda_validation_enabled',
            'lambda_validation_per_class', 'lambda_validation_split_seed',
        )
        for dataset in baseline.DATASETS:
            with self.subTest(dataset=dataset):
                adaptive = options(command_for(
                    dataset, 42, python=Path('/python'),
                    worktree=Path('/code'), results_dir=Path('/results'),
                    data_root=Path('/data'), device='cuda:0'))
                replay = registry._option_map(baseline.command_for(
                    (dataset, 'er', 42), Path('/results'),
                    python=Path('/python'), worktree=Path('/code'),
                    device='cuda:0', data_root=Path('/data')), start=2)
                self.assertEqual(
                    {key: adaptive.get(key) for key in shared},
                    {key: replay.get(key) for key in shared})
                self.assertEqual(adaptive['cl_method'], 'proto_evolve')
                self.assertEqual(adaptive['head_consolidation_mode'],
                                 'adaptive_dual_branch')
                self.assertEqual(adaptive['formal_deferred_evaluation'], '1')

    def test_upmc_uses_four_party_cache(self):
        command = command_for(
            'upmc_food101', 42, python=Path('/python'),
            worktree=Path('/code'), results_dir=Path('/results'),
            data_root=Path('/data'), device='cuda:0')
        actual = options(command)
        self.assertEqual(actual['num_parties'], '4')
        self.assertEqual(actual['vector_npz'],
                         str(Path('/data') / baseline.UPMC_FOUR_CACHE))


if __name__ == '__main__':
    unittest.main()
