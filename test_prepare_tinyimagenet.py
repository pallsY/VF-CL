import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_tinyimagenet import (
    HELDOUT_SPLIT_SEED,
    build_heldout_manifest,
    frozen_protocol,
    reorganize_validation,
    split_train_validation,
    validate,
)


class PrepareTinyImageNetTests(unittest.TestCase):
    def test_reorganize_validation_uses_annotation_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'tiny-imagenet-200'
            images = root / 'val' / 'images'
            images.mkdir(parents=True)
            (images / 'sample.JPEG').write_bytes(b'image')
            (root / 'val' / 'val_annotations.txt').write_text(
                'sample.JPEG\tn00000001\t0\t0\t1\t1\n'
            )
            reorganize_validation(root)
            self.assertTrue((root / 'val' / 'n00000001' / 'sample.JPEG').is_file())
            self.assertFalse(images.exists())

    def test_validate_rejects_incomplete_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            (root / 'val').mkdir()
            (root / 'wnids.txt').write_text('n00000001\n')
            with self.assertRaisesRegex(ValueError, '200 matching'):
                validate(root)

    def test_frozen_protocol_has_exact_heldout_boundaries(self):
        protocol = frozen_protocol()
        self.assertEqual(HELDOUT_SPLIT_SEED, 20260813)
        self.assertEqual(protocol['classes'], 200)
        self.assertEqual(protocol['tasks'], 10)
        self.assertEqual(protocol['classes_per_task'], 20)
        self.assertEqual(protocol['train_per_class'], 450)
        self.assertEqual(protocol['validation_per_class'], 50)
        self.assertEqual(protocol['official_val_test_per_class'], 50)
        self.assertEqual(protocol['parties'], 4)
        self.assertEqual(protocol['party_widths'], [16, 16, 16, 16])
        self.assertEqual(protocol['model'], 'resnet18')
        self.assertEqual(protocol['aggregation'], 'sum')
        self.assertEqual(protocol['train_transform'], [
            'RandomCrop(64,8)', 'RandomHorizontalFlip', 'ToTensor',
            'Normalize([0.480,0.448,0.398],[0.277,0.269,0.282])',
        ])
        self.assertEqual(protocol['validation_test_transform'], [
            'ToTensor',
            'Normalize([0.480,0.448,0.398],[0.277,0.269,0.282])',
        ])
        self.assertEqual(protocol['test_source'], 'official_labeled_validation')
        self.assertEqual(protocol['unlabeled_test_access'], False)

    def test_split_is_deterministic_path_based_and_disjoint(self):
        samples = {
            'n00000001': [f'n00000001/images/{index}.JPEG' for index in range(5)],
            'n00000002': [f'n00000002/images/{index}.JPEG' for index in range(5)],
        }
        first = split_train_validation(samples, per_class=2)
        second = split_train_validation({
            key: list(reversed(value)) for key, value in reversed(list(samples.items()))
        }, per_class=2)
        self.assertEqual(first, second)
        self.assertEqual(first['seed'], 20260813)
        self.assertEqual(len(first['validation_paths']), 4)
        self.assertEqual(len(first['training_paths']), 6)
        self.assertFalse(set(first['validation_paths']) & set(first['training_paths']))
        self.assertRegex(first['sha256'], r'^[0-9a-f]{64}$')

    def test_manifest_hashes_archive_paths_classes_tasks_transforms_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = ['n00000001', 'n00000002']
            (root / 'wnids.txt').write_text('\n'.join(classes) + '\n')
            (root / 'DOWNLOAD_SHA256.txt').write_text('a' * 64 + '  tiny-imagenet-200.zip\n')
            for class_name in classes:
                train = root / 'train' / class_name / 'images'
                val = root / 'val' / class_name
                train.mkdir(parents=True)
                val.mkdir(parents=True)
                for index in range(5):
                    (train / f'{index}.JPEG').write_bytes(f'{class_name}:{index}'.encode())
                for index in range(2):
                    (val / f'{index}.JPEG').write_bytes(f'val:{class_name}:{index}'.encode())
            manifest = build_heldout_manifest(
                root, expected_classes=2, train_per_class=5,
                validation_per_class=2, official_val_per_class=2,
            )
            self.assertEqual(manifest['archive_sha256'], 'a' * 64)
            self.assertEqual(manifest['class_order'], classes)
            self.assertEqual(manifest['task_manifest'], [classes])
            self.assertEqual(len(manifest['source_file_sha256']), 14)
            self.assertEqual(len(manifest['official_val_test_paths']), 4)
            self.assertTrue(all(
                not path.startswith('train/')
                for path in manifest['training_validation_paths']
            ))
            self.assertNotIn('/test/', json.dumps(manifest))
            for key in ('paths_sha256', 'class_order_sha256',
                        'task_manifest_sha256', 'transforms_sha256',
                        'split_sha256', 'manifest_sha256'):
                self.assertRegex(manifest[key], r'^[0-9a-f]{64}$')
            payload = dict(manifest)
            digest = payload.pop('manifest_sha256')
            encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
            self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())


if __name__ == '__main__':
    unittest.main()
