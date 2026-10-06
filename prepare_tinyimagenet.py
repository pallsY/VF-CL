#!/usr/bin/env python3
"""Download, verify, and normalize the official TinyImageNet-200 layout."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from calibration_split import build_manifest


URL = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
HELDOUT_SPLIT_SEED = 20260813


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def frozen_protocol():
    return {
        'classes': 200, 'tasks': 10, 'classes_per_task': 20,
        'train_per_class': 450, 'validation_per_class': 50,
        'official_val_test_per_class': 50,
        'validation_split_seed': HELDOUT_SPLIT_SEED,
        'parties': 4, 'party_widths': [16, 16, 16, 16],
        'model': 'resnet18', 'aggregation': 'sum',
        'train_transform': [
            'RandomCrop(64,8)', 'RandomHorizontalFlip', 'ToTensor',
            'Normalize([0.480,0.448,0.398],[0.277,0.269,0.282])',
        ],
        'validation_test_transform': [
            'ToTensor',
            'Normalize([0.480,0.448,0.398],[0.277,0.269,0.282])',
        ],
        'test_source': 'official_labeled_validation',
        'unlabeled_test_access': False,
    }


def split_train_validation(samples_by_class, per_class=50,
                           seed=HELDOUT_SPLIT_SEED):
    classes = sorted(samples_by_class)
    sample_ids = []
    targets = []
    for class_id, class_name in enumerate(classes):
        paths = sorted(str(path) for path in samples_by_class[class_name])
        if len(paths) < per_class:
            raise ValueError(f'class {class_name} has only {len(paths)} train samples')
        sample_ids.extend(paths)
        targets.extend([class_id] * len(paths))
    manifest = build_manifest(
        targets, per_class=per_class, seed=seed,
        dataset='tinyimagenet-train', sample_ids=sample_ids,
    )
    selected = manifest['ordered_sample_ids']
    validation = sorted(selected)
    selected_set = set(validation)
    training = sorted(path for path in sample_ids if path not in selected_set)
    payload = {
        'seed': int(seed), 'per_class': int(per_class),
        'training_paths': training, 'validation_paths': validation,
    }
    return {**payload, 'sha256': manifest['sha256'],
            'logical_sha256': _canonical_sha256(payload)}


def build_heldout_manifest(root, expected_classes=200, train_per_class=500,
                           validation_per_class=50,
                           official_val_per_class=50):
    root = Path(root).resolve()
    words = sorted(line.strip() for line in (root / 'wnids.txt').read_text().splitlines()
                   if line.strip())
    if len(words) != expected_classes or len(set(words)) != expected_classes:
        raise ValueError(f'TinyImageNet requires {expected_classes} unique classes')
    archive_tokens = (root / 'DOWNLOAD_SHA256.txt').read_text().split()
    if not archive_tokens or len(archive_tokens[0]) != 64:
        raise ValueError('TinyImageNet archive SHA-256 is missing')
    archive_sha256 = archive_tokens[0].lower()
    if any(character not in '0123456789abcdef' for character in archive_sha256):
        raise ValueError('TinyImageNet archive SHA-256 is malformed')
    train_by_class = {}
    official_val = []
    source_hashes = {}
    for class_name in words:
        train = sorted((root / 'train' / class_name / 'images').glob('*.JPEG'))
        val = sorted((root / 'val' / class_name).glob('*.JPEG'))
        if len(train) != train_per_class or len(val) != official_val_per_class:
            raise ValueError(f'TinyImageNet class count mismatch: {class_name}')
        train_ids = [path.relative_to(root / 'train').as_posix() for path in train]
        val_ids = [path.relative_to(root).as_posix() for path in val]
        train_by_class[class_name] = train_ids
        official_val.extend(val_ids)
        for path in (*train, *val):
            source_hashes[path.relative_to(root).as_posix()] = sha256(path)
    split = split_train_validation(
        train_by_class, per_class=validation_per_class,
    )
    task_width = 20 if expected_classes == 200 else expected_classes
    tasks = [words[start:start + task_width]
             for start in range(0, expected_classes, task_width)]
    protocol = frozen_protocol()
    paths = sorted(source_hashes)
    payload = {
        'schema_version': 1, 'dataset': 'tinyimagenet-200-heldout',
        'dataset_root': str(root),
        'archive_sha256': archive_sha256,
        'source_file_sha256': source_hashes,
        'class_order': words, 'task_manifest': tasks,
        'training_paths': split['training_paths'],
        'training_validation_paths': split['validation_paths'],
        'official_val_test_paths': sorted(official_val),
        'protocol': protocol,
        'paths_sha256': _canonical_sha256(paths),
        'class_order_sha256': _canonical_sha256(words),
        'task_manifest_sha256': _canonical_sha256(tasks),
        'transforms_sha256': _canonical_sha256({
            'train': protocol['train_transform'],
            'validation_test': protocol['validation_test_transform'],
        }),
        'split_sha256': split['sha256'],
    }
    return {**payload, 'manifest_sha256': _canonical_sha256(payload)}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(root):
    root = Path(root)
    words = [line.strip() for line in (root / 'wnids.txt').read_text().splitlines()
             if line.strip()]
    train = {path.name for path in (root / 'train').iterdir() if path.is_dir()}
    val = {path.name for path in (root / 'val').iterdir() if path.is_dir()}
    if len(words) != 200 or train != set(words) or val != set(words):
        raise ValueError('TinyImageNet must have 200 matching train/val classes')
    train_images = sum(1 for _ in (root / 'train').glob('*/images/*.JPEG'))
    val_images = sum(1 for _ in (root / 'val').glob('*/*.JPEG'))
    if train_images != 100000 or val_images != 10000:
        raise ValueError(
            f'TinyImageNet image count mismatch: train={train_images}, val={val_images}'
        )
    return {'classes': 200, 'train_images': train_images, 'val_images': val_images}


def reorganize_validation(root):
    root = Path(root).resolve()
    val = (root / 'val').resolve()
    images = (val / 'images').resolve()
    if root not in images.parents or not images.is_dir():
        return
    annotations = {}
    for line in (val / 'val_annotations.txt').read_text().splitlines():
        name, class_id, *_ = line.split('\t')
        annotations[name] = class_id
    for name, class_id in annotations.items():
        source = (images / name).resolve()
        target_dir = (val / class_id).resolve()
        if root not in source.parents or root not in target_dir.parents:
            raise ValueError('unsafe TinyImageNet validation path')
        target_dir.mkdir(exist_ok=True)
        shutil.move(str(source), str(target_dir / name))
    images.rmdir()


def prepare(data_path):
    data_path = Path(data_path).resolve()
    root = data_path / 'tiny-imagenet-200'
    if root.is_dir():
        return validate(root)
    archive = data_path / 'tiny-imagenet-200.zip.part'
    data_path.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, archive)
    digest = sha256(archive)
    with zipfile.ZipFile(archive) as bundle:
        if any(Path(name).parts[0] != 'tiny-imagenet-200' for name in bundle.namelist()):
            raise ValueError('archive contains an unexpected top-level path')
        bundle.extractall(data_path)
    reorganize_validation(root)
    result = validate(root)
    (root / 'DOWNLOAD_SHA256.txt').write_text(
        f'{digest}  tiny-imagenet-200.zip\n', encoding='utf-8'
    )
    archive.unlink()
    return {**result, 'sha256': digest, 'source': URL}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    args = parser.parse_args()
    print(prepare(args.data_path))


if __name__ == '__main__':
    main()
