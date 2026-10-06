"""Prepare deterministic vector-VFL caches for ISOLET and UPMC Food-101."""
import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


ROOT = Path('/home/chase/Yangxx/VF-CL/data')


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def write_metadata(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _read_isolet_z(path):
    completed = subprocess.run(
        ['gzip', '-dc', str(path)], check=True, stdout=subprocess.PIPE,
    )
    values = np.loadtxt(io.BytesIO(completed.stdout), delimiter=',', dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 618:
        raise ValueError(f'unexpected ISOLET shape from {path}: {values.shape}')
    return values[:, :-1], values[:, -1].astype(np.int64) - 1


def prepare_isolet(force=False):
    from sklearn.preprocessing import StandardScaler

    source = ROOT / 'isolet'
    output = source / 'isolet_vfl.npz'
    metadata = source / 'isolet_vfl.metadata.json'
    if output.is_file() and metadata.is_file() and not force:
        print(f'ISOLET cache exists: {output}')
        return output

    train_file = source / 'isolet1+2+3+4.data.Z'
    test_file = source / 'isolet5.data.Z'
    x_train, y_train = _read_isolet_z(train_file)
    x_test, y_test = _read_isolet_z(test_file)
    if x_train.shape != (6238, 617) or x_test.shape != (1559, 617):
        raise ValueError(f'unexpected official ISOLET split: {x_train.shape}, {x_test.shape}')
    if sorted(np.unique(np.concatenate([y_train, y_test])).tolist()) != list(range(26)):
        raise ValueError('ISOLET labels must cover 0..25 after normalization')

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    train_idx = np.arange(len(x_train), dtype=np.int64)
    test_idx = np.arange(len(x_train), len(x), dtype=np.int64)
    boundaries = [0, 155, 309, 463, 617]
    view_names = np.asarray([f'acoustic_block_{i}' for i in range(4)])
    range_lo = np.asarray(boundaries[:-1], dtype=np.int64)
    range_hi = np.asarray(boundaries[1:], dtype=np.int64)

    tmp = output.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as handle:
        np.savez_compressed(
            handle, X=x, y=y, train_idx=train_idx, test_idx=test_idx,
            view_names=view_names, range_lo=range_lo, range_hi=range_hi,
        )
    os.replace(tmp, output)
    write_metadata(metadata, {
        'dataset': 'ISOLET',
        'official_train_files': [train_file.name],
        'official_test_files': [test_file.name],
        'source_sha256': {
            train_file.name: sha256_file(train_file),
            test_file.name: sha256_file(test_file),
        },
        'train_count': len(x_train),
        'test_count': len(x_test),
        'num_classes': 26,
        'num_features': 617,
        'views': [
            {'name': str(name), 'lo': int(lo), 'hi': int(hi)}
            for name, lo, hi in zip(view_names, range_lo, range_hi)
        ],
        'standardization_fit': 'official training split only',
        'output_sha256': sha256_file(output),
    })
    print(f'ISOLET ready: {output} X={x.shape}')
    return output


def _read_upmc_csv(path):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'id', 'text', 'annotation', 'label'}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f'{path}: expected columns {sorted(required)}, got {reader.fieldnames}')
        for row in reader:
            rows.append(row)
    return rows


def _clean_title(value):
    value = html.unescape(value or '').lower()
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return ' '.join(value.split())


def _load_resnet18_feature_model(device, weights_path):
    import torch
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


class _UPMCImageDataset:
    def __init__(self, rows, image_root, split, transform):
        self.rows = rows
        self.image_root = image_root
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        path = self.image_root / self.split / row['annotation'] / row['id']
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, index


def _extract_image_features(rows, image_root, split, model, device, batch_size, workers):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = _UPMCImageDataset(rows, image_root, split, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.startswith('cuda'), persistent_workers=workers > 0,
    )
    output = np.empty((len(rows), 512), dtype=np.float32)
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader):
            features = model(images.to(device, non_blocking=True))
            features = torch.nn.functional.normalize(features, dim=1)
            output[indices.numpy()] = features.cpu().numpy().astype(np.float32)
            if batch_index % 50 == 0:
                print(f'UPMC {split}: {min((batch_index + 1) * batch_size, len(rows))}/{len(rows)}')
    return output


def prepare_upmc(force=False, device='cuda:0', batch_size=256, workers=4):
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer

    source = ROOT / 'UPMC_Food101' / 'UPMC-Food-101'
    output_dir = ROOT / 'upmc_food101'
    output = output_dir / 'upmc_food101_vfl.npz'
    metadata = output_dir / 'upmc_food101_vfl.metadata.json'
    if output.is_file() and metadata.is_file() and not force:
        print(f'UPMC cache exists: {output}')
        return output
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for UPMC feature extraction but unavailable')

    train_csv = source / 'train.csv'
    test_csv = source / 'test.csv'
    train_rows = _read_upmc_csv(train_csv)
    test_rows = _read_upmc_csv(test_csv)
    all_rows = train_rows + test_rows
    labels = np.asarray([int(row['label']) for row in all_rows], dtype=np.int64)
    if sorted(np.unique(labels).tolist()) != list(range(101)):
        raise ValueError('UPMC labels must cover 0..100')
    duplicate_ids = len(all_rows) - len({(row['annotation'], row['id']) for row in all_rows})
    if duplicate_ids:
        raise ValueError(f'UPMC contains {duplicate_ids} duplicate annotation/id pairs')

    weights_path = Path('/home/chase/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    if not weights_path.is_file():
        raise FileNotFoundError(f'local ResNet-18 weights missing: {weights_path}')
    model = _load_resnet18_feature_model(device, weights_path)
    image_root = source / 'images'
    image_train = _extract_image_features(
        train_rows, image_root, 'train', model, device, batch_size, workers,
    )
    image_test = _extract_image_features(
        test_rows, image_root, 'test', model, device, batch_size, workers,
    )

    train_titles = [_clean_title(row['text']) for row in train_rows]
    test_titles = [_clean_title(row['text']) for row in test_rows]
    vectorizer = TfidfVectorizer(
        max_features=1024, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        strip_accents='unicode', dtype=np.float32,
    )
    text_train = vectorizer.fit_transform(train_titles).toarray().astype(np.float32)
    text_test = vectorizer.transform(test_titles).toarray().astype(np.float32)
    if text_train.shape[1] != 1024:
        raise ValueError(f'expected 1024 TF-IDF features, got {text_train.shape[1]}')

    x_train = np.concatenate([image_train, text_train], axis=1)
    x_test = np.concatenate([image_test, text_test], axis=1)
    x = np.concatenate([x_train, x_test], axis=0)
    train_idx = np.arange(len(train_rows), dtype=np.int64)
    test_idx = np.arange(len(train_rows), len(all_rows), dtype=np.int64)
    view_names = np.asarray(['image_resnet18_imagenet1k', 'title_tfidf_train_only'])
    range_lo = np.asarray([0, 512], dtype=np.int64)
    range_hi = np.asarray([512, 1536], dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as handle:
        np.savez_compressed(
            handle, X=x, y=labels, train_idx=train_idx, test_idx=test_idx,
            view_names=view_names, range_lo=range_lo, range_hi=range_hi,
        )
    os.replace(tmp, output)
    vocabulary_hash = hashlib.sha256(
        '\n'.join(sorted(vectorizer.vocabulary_)).encode('utf-8')
    ).hexdigest()
    write_metadata(metadata, {
        'dataset': 'UPMC Food-101',
        'official_split': 'train.csv/test.csv',
        'source_sha256': {
            'train.csv': sha256_file(train_csv),
            'test.csv': sha256_file(test_csv),
        },
        'train_count': len(train_rows),
        'test_count': len(test_rows),
        'num_classes': 101,
        'num_features': 1536,
        'views': [
            {'name': 'image_resnet18_imagenet1k', 'lo': 0, 'hi': 512},
            {'name': 'title_tfidf_train_only', 'lo': 512, 'hi': 1536},
        ],
        'image_encoder': 'frozen torchvision ResNet-18 ImageNet-1K',
        'image_weights_sha256': sha256_file(weights_path),
        'text_encoder': 'TF-IDF 1-2 grams, fit on training titles only',
        'text_vocabulary_sha256': vocabulary_hash,
        'test_used_for_fit': False,
        'output_sha256': sha256_file(output),
    })
    print(f'UPMC ready: {output} X={x.shape}')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['isolet', 'upmc', 'all'], default='all')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.dataset in ('isolet', 'all'):
        prepare_isolet(args.force)
    if args.dataset in ('upmc', 'all'):
        prepare_upmc(args.force, args.device, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
