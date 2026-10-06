import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from calibration_split import build_manifest, manifest_indices
import data_utils
from data_utils import VFLDataset
from fair_main_table_3datasets import DATASETS as FAIR_DATASETS
from unified_head_consolidation_factorial import (
    EXPECTED_VALIDATION_HASHES,
    EXPECTED_VALIDATION_LABELS,
)


class TinyDataset(Dataset):
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([float(index)]), self.targets[index]


class FakeImageFolder(Dataset):
    calls = []
    reverse_train = False

    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.classes = [f'n{class_id:08d}' for class_id in range(200)]
        per_class = 500 if path.endswith('train') else 1
        samples = [
            (
                os.path.join(
                    path, class_name, 'images',
                    (f'{class_name}_{sample_id:03d}.JPEG'
                     if path.endswith('train')
                     else f'val_{class_name}_{sample_id:03d}.JPEG'),
                ),
                class_id,
            )
            for class_id, class_name in enumerate(self.classes)
            for sample_id in range(per_class)
        ]
        if path.endswith('train') and self.reverse_train:
            samples.reverse()
        self.samples = samples
        self.targets = [label for _, label in samples]
        type(self).calls.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.tensor([float(index)]), self.targets[index]


class FrozenAdaptiveValidationSplitTest(unittest.TestCase):
    def test_cifar_validation_identity_remains_frozen(self):
        targets = [class_id for class_id in range(100) for _ in range(500)]
        calibration = build_manifest(
            targets, per_class=25, seed=20260722,
        )
        validation = build_manifest(
            targets,
            per_class=25,
            seed=20260729,
            dataset='cifar100-train',
            excluded_indices=manifest_indices(calibration),
        )

        self.assertEqual(validation['dataset'], 'cifar100-train')
        self.assertEqual(validation['seed'], 20260729)
        self.assertEqual(validation['per_class'], 25)
        self.assertEqual(len(validation['ordered_indices']), 2500)
        self.assertEqual(
            validation['sha256'],
            'a47ebde3ff03cd4253d4a8a272948dea6229edbbc8be6b96f574c344b48604c8',
        )
        self.assertEqual(
            EXPECTED_VALIDATION_HASHES['cifar100'],
            '0aa4729ade65021ce774c757516584c4d2a2ce70d2879b045db47dbceae917fa',
        )

    def test_vector_validation_contracts_remain_frozen(self):
        contracts = {
            'isolet': {
                'dataset': 'isolet_vfl.npz-train',
                'seed': 20260809,
                'per_class': 40,
                'classes': 26,
                'sha256': '487e81663a12d4a663a1f421407aa3f88d788cc3c83f7323166cf7aff82902d3',
            },
            'upmc_food101': {
                'dataset': 'upmc_food101_vfl.npz-train',
                'seed': 20260809,
                'per_class': 64,
                'classes': 101,
                'sha256': 'b812e99da856d94ee5b6eea28eadaf9924f52e355d44941c42da4028c0895e06',
            },
        }
        self.assertEqual(contracts['isolet']['classes'] * contracts['isolet']['per_class'], 1040)
        self.assertEqual(
            contracts['upmc_food101']['classes'] * contracts['upmc_food101']['per_class'],
            6464,
        )
        self.assertEqual(
            {key: value['dataset'] for key, value in contracts.items()},
            {
                'isolet': 'isolet_vfl.npz-train',
                'upmc_food101': 'upmc_food101_vfl.npz-train',
            },
        )
        self.assertEqual(
            {key: EXPECTED_VALIDATION_HASHES[key] for key in contracts},
            {key: value['sha256'] for key, value in contracts.items()},
        )
        self.assertEqual(
            {key: EXPECTED_VALIDATION_LABELS[key] for key in contracts},
            {key: value['dataset'] for key, value in contracts.items()},
        )
        for dataset, contract in contracts.items():
            self.assertEqual(FAIR_DATASETS[dataset]['num_classes'], contract['classes'])
            self.assertEqual(
                FAIR_DATASETS[dataset]['validation_per_class'],
                contract['per_class'],
            )
            self.assertEqual(
                FAIR_DATASETS[dataset]['validation_split_seed'],
                contract['seed'],
            )

    def test_tinyimagenet_uses_stable_relative_path_holdout_and_official_val_test(self):
        self.assertIn('sample_ids', inspect.signature(build_manifest).parameters)
        self.assertTrue(hasattr(data_utils, 'relative_sample_path'))
        relative_sample_path = data_utils.relative_sample_path
        FakeImageFolder.calls = []
        args = SimpleNamespace(
            data='tinyimagenet',
            data_path='/fixture',
            bic_enabled=0,
            lambda_validation_enabled=1,
            lambda_validation_per_class=50,
            lambda_validation_split_seed=20260813,
            output_dir=None,
            deterministic=0,
            data_flow_audit=0,
            batch_size=32,
            num_workers=0,
        )
        with tempfile.TemporaryDirectory() as output_dir, patch(
                'data_utils.datasets.ImageFolder', FakeImageFolder):
            args.output_dir = output_dir
            FakeImageFolder.reverse_train = False
            first = VFLDataset(args)
            FakeImageFolder.reverse_train = True
            second = VFLDataset(args)

        self.assertEqual(first.validation_manifest, second.validation_manifest)
        manifest = first.validation_manifest
        self.assertEqual(manifest['dataset'], 'tinyimagenet-train')
        self.assertEqual(manifest['seed'], 20260813)
        self.assertEqual(manifest['per_class'], 50)
        self.assertEqual(len(manifest['ordered_sample_ids']), 10000)
        self.assertNotIn('ordered_indices', manifest)
        self.assertTrue(all(
            len(manifest['by_class'][str(class_id)]) == 50
            for class_id in range(200)
        ))

        selected = set(manifest['ordered_sample_ids'])
        all_train = {
            relative_sample_path(path, first.trainset.path)
            for path, _ in first.trainset.samples
        }
        training = all_train - selected
        official_val = {
            relative_sample_path(path, first.testset.path)
            for path, _ in first.testset.samples
        }
        self.assertEqual(len(training), 90000)
        for class_id in range(200):
            held_out = sum(
                first.trainset.targets[index] == class_id
                for index in first.validation_indices
            )
            self.assertEqual(held_out, 50)
            self.assertEqual(first.trainset.targets.count(class_id) - held_out, 450)
        self.assertFalse(training & selected)
        self.assertFalse((training | selected) & official_val)
        self.assertTrue(first.testset.path.endswith(os.path.join('tiny-imagenet-200', 'val')))
        self.assertFalse(any(path.endswith(os.path.join('tiny-imagenet-200', 'test'))
                             for path in FakeImageFolder.calls))
        self.assertEqual(
            first.selection_audit()['evaluation_source'],
            'tinyimagenet-train-validation',
        )

    def test_explicit_loaders_keep_training_validation_and_test_disjoint(self):
        dataset = VFLDataset.__new__(VFLDataset)
        self.assertTrue(hasattr(dataset, 'get_train_loader'))
        self.assertTrue(hasattr(dataset, 'get_test_loader'))
        dataset.args = SimpleNamespace(
            bic_enabled=1, lambda_validation_enabled=1, deterministic=0,
            data_flow_audit=0, batch_size=8, num_workers=0,
        )
        dataset.trainset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.validationset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.testset = TinyDataset([0, 1])
        dataset.calibration_indices = {0, 3}
        dataset.validation_indices = {1, 4}

        train = dataset.get_train_loader([0, 1], shuffle=False)
        validation = dataset.get_validation_loader([0, 1])
        test = dataset.get_test_loader([0, 1])

        self.assertEqual(train.dataset.indices, [2, 5])
        self.assertEqual(validation.dataset.indices, [1, 4])
        self.assertEqual(test.dataset.indices, [0, 1])
        legacy_train, legacy_evaluation = dataset.get_task_loaders(
            [0, 1], shuffle_train=False,
        )
        self.assertEqual(legacy_train.dataset.indices, [2, 5])
        self.assertEqual(legacy_evaluation.dataset.indices, [1, 4])


if __name__ == '__main__':
    unittest.main()
