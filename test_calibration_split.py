import json
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

import calibration_split
from calibration_split import build_manifest, manifest_indices, write_manifest
from data_utils import VFLDataset


class TinyDataset(Dataset):
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor([float(index)]), self.targets[index]


class CalibrationSplitTests(unittest.TestCase):
    def test_manifest_records_explicit_dataset_name(self):
        manifest = build_manifest(
            [0] * 3 + [1] * 3, per_class=2, seed=7,
            dataset='tinyimagenet-train',
        )
        self.assertEqual(manifest['dataset'], 'tinyimagenet-train')

    def test_tinyimagenet_bic_uses_separate_clean_calibration_view(self):
        class FakeImageFolder:
            instances = []

            def __init__(self, path, transform):
                self.path, self.transform = path, transform
                self.classes = [f'n{value:08d}' for value in range(200)]
                per_class = 30 if path.endswith('train') else 5
                self.targets = [label for label in range(200)
                                for _ in range(per_class)]
                FakeImageFolder.instances.append(self)

        with tempfile.TemporaryDirectory() as output_dir, patch(
                'data_utils.datasets.ImageFolder', FakeImageFolder):
            args = SimpleNamespace(
                data='tinyimagenet', data_path='/unused', bic_enabled=1,
                bic_per_class=25, bic_split_seed=20260722,
                output_dir=output_dir,
            )
            dataset = VFLDataset(args)

        self.assertIsNot(dataset.trainset, dataset.calibrationset)
        self.assertEqual(dataset.calibration_manifest['dataset'],
                         'tinyimagenet-train')
        self.assertEqual(len(dataset.calibration_indices), 5000)
        self.assertEqual(dataset.trainset.classes, dataset.testset.classes)

    def setUp(self):
        self.targets = [class_id for class_id in range(100) for _ in range(500)]

    def test_manifest_is_deterministic_balanced_and_unique(self):
        first = build_manifest(self.targets, per_class=25, seed=20260722)
        second = build_manifest(self.targets, per_class=25, seed=20260722)

        self.assertEqual(first, second)
        self.assertEqual(len(manifest_indices(first)), 2500)
        self.assertEqual(list(first), [
            'dataset', 'seed', 'per_class', 'by_class',
            'ordered_indices', 'sha256',
        ])
        self.assertEqual(len(first['sha256']), 64)
        for class_id in range(100):
            selected = first['by_class'][str(class_id)]
            self.assertEqual(len(selected), 25)
            self.assertEqual(len(set(selected)), 25)
            self.assertTrue(all(self.targets[index] == class_id for index in selected))

    def test_manifest_rejects_insufficient_class_samples(self):
        with self.assertRaisesRegex(ValueError, 'class 1 has 1 samples'):
            build_manifest([0, 0, 1], per_class=2, seed=20260722)

    def test_manifest_excludes_reserved_indices(self):
        calibration = build_manifest(
            self.targets, per_class=25, seed=20260722
        )
        reserved = manifest_indices(calibration)
        validation = build_manifest(
            self.targets, per_class=25, seed=20260729,
            excluded_indices=reserved,
        )

        self.assertFalse(reserved & manifest_indices(validation))

    def test_write_manifest_round_trips_exactly(self):
        manifest = build_manifest(self.targets, per_class=25, seed=20260722)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_manifest(manifest, path)
            self.assertEqual(json.loads(path.read_text()), manifest)

    def test_optional_sample_ids_are_enumeration_independent(self):
        self.assertIn('sample_ids', inspect.signature(build_manifest).parameters)
        targets = [0, 0, 0, 1, 1, 1]
        sample_ids = ['a/3', 'a/1', 'a/2', 'b/2', 'b/3', 'b/1']
        first = build_manifest(
            targets, per_class=2, seed=17, sample_ids=sample_ids,
        )
        order = [5, 0, 3, 2, 1, 4]
        second = build_manifest(
            [targets[index] for index in order],
            per_class=2,
            seed=17,
            sample_ids=[sample_ids[index] for index in order],
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first['ordered_sample_ids']), {'a/2', 'a/3', 'b/1', 'b/2'})

    def test_write_manifest_fsyncs_file_replaces_atomically_and_fsyncs_parent(self):
        self.assertTrue(hasattr(calibration_split, 'os'))
        manifest = build_manifest(self.targets, per_class=25, seed=20260722)
        expected = json.dumps(manifest, indent=2, sort_keys=True).encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            with patch('calibration_split.os.replace', wraps=os.replace) as replace, patch(
                    'calibration_split.os.fsync', wraps=os.fsync) as fsync:
                write_manifest(manifest, path)

            self.assertEqual(path.read_bytes(), expected)
            replace.assert_called_once()
            self.assertEqual(Path(replace.call_args.args[1]), path)
            self.assertGreaterEqual(fsync.call_count, 2)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_training_loader_excludes_calibration_indices(self):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.args = SimpleNamespace(
            bic_enabled=1, lambda_validation_enabled=0, deterministic=0,
            batch_size=8, num_workers=0
        )
        dataset.trainset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.testset = TinyDataset([0, 1])
        dataset.calibrationset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.calibration_indices = {0, 3}

        train_loader, _ = dataset.get_task_loaders([0, 1], shuffle_train=False)
        calibration_loader = dataset.get_calibration_loader([0, 1])

        self.assertEqual(train_loader.dataset.indices, [1, 2, 4, 5])
        self.assertEqual(calibration_loader.dataset.indices, [0, 3])

    def test_validation_mode_excludes_both_holdouts_and_routes_evaluation(self):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.args = SimpleNamespace(
            bic_enabled=1, lambda_validation_enabled=1, deterministic=0,
            batch_size=8, num_workers=0,
        )
        dataset.trainset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.testset = TinyDataset([0, 1])
        dataset.calibrationset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.validationset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.calibration_indices = {0, 3}
        dataset.validation_indices = {1, 4}

        train_loader, evaluation_loader = dataset.get_task_loaders(
            [0, 1], shuffle_train=False
        )
        validation_loader = dataset.get_validation_loader([0, 1])

        self.assertEqual(train_loader.dataset.indices, [2, 5])
        self.assertIs(evaluation_loader.dataset.dataset, dataset.validationset)
        self.assertEqual(evaluation_loader.dataset.indices, [1, 4])
        self.assertEqual(validation_loader.dataset.indices, [1, 4])

    def test_calibration_audit_reports_disjoint_sets(self):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.trainset = TinyDataset([0, 0, 1, 1])
        dataset.calibration_indices = {0, 2}
        dataset.calibration_manifest = {'sha256': 'a' * 64, 'per_class': 1}
        audit = dataset.calibration_audit()
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['calibration_count'], 2)

    def test_selection_audit_reports_pairwise_disjoint_sets(self):
        dataset = VFLDataset.__new__(VFLDataset)
        dataset.trainset = TinyDataset([0, 0, 0, 1, 1, 1])
        dataset.calibration_indices = {0, 3}
        dataset.validation_indices = {1, 4}
        dataset.calibration_manifest = {'sha256': 'a' * 64, 'per_class': 1}
        dataset.validation_manifest = {'sha256': 'b' * 64, 'per_class': 1}

        audit = dataset.selection_audit()

        self.assertTrue(audit['passed'])
        self.assertEqual(audit['training_count'], 2)
        self.assertEqual(audit['calibration_validation_overlap_count'], 0)
        self.assertFalse(audit['test_used_for_selection'])
        self.assertEqual(audit['evaluation_source'], 'cifar100-train-validation')


if __name__ == '__main__':
    unittest.main()
