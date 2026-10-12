"""Build a separate UPMC cache from a frozen image-clean selection manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from prepare_fair_datasets import (
    _clean_title, _extract_image_features, _load_resnet18_feature_model,
    _read_upmc_csv, sha256_file,
)
from upmc_image_clean_split import (
    EXPECTED_COUNTS, EXPECTED_SOURCE, build_selection_manifest, select_rows,
)


OUTPUT_NAME = 'upmc_food101_vfl.npz'
METADATA_NAME = 'upmc_food101_vfl.metadata.json'
WEIGHTS = Path('/home/chase/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')


def load_selected_rows(source: Path, manifest: dict):
    """Revalidate the raw source, then return rows and immutable source IDs."""
    source = Path(source)
    archive = (source.parent / 'UPMC-Food-101.zip'
               if 'UPMC-Food-101.zip' in manifest.get('source_sha256', {})
               else None)
    if build_selection_manifest(source, archive) != manifest:
        raise ValueError('source CSV or image bytes differ from frozen selection')
    train_all = _read_upmc_csv(source / 'train.csv')
    test_all = _read_upmc_csv(source / 'test.csv')
    train_indices = select_rows(manifest, 'train')
    test_indices = select_rows(manifest, 'test')
    train = [train_all[index] for index in train_indices]
    test = [test_all[index] for index in test_indices]
    source_ids = np.asarray(
        train_indices + [len(train_all) + index for index in test_indices],
        dtype=np.int64,
    )
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError('selected source identities are not unique')
    return train, test, source_ids


def _existing_cache(output, metadata_path, manifest):
    if output.is_symlink() or metadata_path.is_symlink():
        raise ValueError('image-clean cache symlink is forbidden')
    if not output.exists() and not metadata_path.exists():
        return False
    if not output.is_file() or not metadata_path.is_file():
        raise FileExistsError('partial image-clean cache already exists')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (metadata.get('selection_manifest_sha256')
            != manifest['manifest_sha256']
            or metadata.get('output_sha256') != sha256_file(output)
            or metadata.get('source_sha256') != manifest['source_sha256']):
        raise FileExistsError('image-clean cache identity differs')
    return True


def prepare(source: Path, manifest_path: Path, output_dir: Path, device='cuda:0'):
    from sklearn.feature_extraction.text import TfidfVectorizer

    source, manifest_path, output_dir = (
        Path(source), Path(manifest_path), Path(output_dir))
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('selection manifest must be a regular file')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('source_sha256') != EXPECTED_SOURCE
            or manifest.get('summary') != EXPECTED_COUNTS):
        raise ValueError('selection source or census differs')
    train_rows, test_rows, source_ids = load_selected_rows(source, manifest)
    if len(train_rows) != 62968 or len(test_rows) != 20389:
        raise ValueError('image-clean split size differs')
    output = output_dir / OUTPUT_NAME
    metadata_path = output_dir / METADATA_NAME
    if _existing_cache(output, metadata_path, manifest):
        return output
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    if device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable for UPMC extraction')
    model = _load_resnet18_feature_model(device, WEIGHTS)
    image_root = source / 'images'
    image_train = _extract_image_features(
        train_rows, image_root, 'train', model, device, 256, 4)
    image_test = _extract_image_features(
        test_rows, image_root, 'test', model, device, 256, 4)
    vectorizer = TfidfVectorizer(
        max_features=1024, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, strip_accents='unicode', dtype=np.float32,
    )
    train_titles = [_clean_title(row['text']) for row in train_rows]
    test_titles = [_clean_title(row['text']) for row in test_rows]
    text_train = vectorizer.fit_transform(train_titles).toarray().astype(np.float32)
    text_test = vectorizer.transform(test_titles).toarray().astype(np.float32)
    if text_train.shape[1] != 1024:
        raise ValueError('UPMC training vocabulary does not have 1024 features')
    x_train = np.concatenate((image_train, text_train), axis=1)
    x_test = np.concatenate((image_test, text_test), axis=1)
    x = np.concatenate((x_train, x_test), axis=0)
    if x.shape != (83357, 1536) or not np.isfinite(x).all():
        raise ValueError('UPMC image-clean feature matrix is invalid')
    labels = np.asarray(
        [int(row['label']) for row in train_rows + test_rows], dtype=np.int64)
    if sorted(np.unique(labels).tolist()) != list(range(101)):
        raise ValueError('UPMC image-clean labels do not cover 0..100')

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=output_dir, prefix='.upmc-image-clean-',
            suffix='.npz.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle, X=x, y=labels,
            train_idx=np.arange(len(train_rows), dtype=np.int64),
            test_idx=np.arange(len(train_rows), len(labels), dtype=np.int64),
            source_row_id=source_ids,
            view_names=np.asarray([
                'image_resnet18_imagenet1k', 'title_tfidf_train_only']),
            range_lo=np.asarray([0, 512], dtype=np.int64),
            range_hi=np.asarray([512, 1536], dtype=np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    try:
        output_hash = sha256_file(temporary)
        metadata = {
            'dataset': 'UPMC Food-101 image-clean v1',
            'selection_manifest_sha256': manifest['manifest_sha256'],
            'selection_manifest_file_sha256': sha256_file(manifest_path),
            'source_sha256': manifest['source_sha256'],
            'train_count': len(train_rows),
            'test_count': len(test_rows),
            'num_classes': 101,
            'num_features': 1536,
            'source_row_id_sha256': hashlib.sha256(
                source_ids.tobytes()).hexdigest(),
            'views': [
                {'name': 'image_resnet18_imagenet1k', 'lo': 0, 'hi': 512},
                {'name': 'title_tfidf_train_only', 'lo': 512, 'hi': 1536},
            ],
            'image_encoder': 'frozen torchvision ResNet-18 ImageNet-1K',
            'image_weights_sha256': sha256_file(WEIGHTS),
            'text_encoder': 'TF-IDF 1-2 grams, fit on selected training titles only',
            'text_vocabulary_sha256': hashlib.sha256(
                '\n'.join(sorted(vectorizer.vocabulary_)).encode('utf-8')
            ).hexdigest(),
            'test_used_for_fit': False,
            'output_sha256': output_hash,
        }
        metadata_bytes = (json.dumps(
            metadata, indent=2, sort_keys=True, allow_nan=False) + '\n'
        ).encode('utf-8')
        # Hard-link installation refuses an existing output without replacing it.
        os.link(temporary, output)
        with metadata_path.open('xb') as handle:
            handle.write(metadata_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        return output
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(prepare(
        args.source, args.manifest, args.output_dir, args.device))


if __name__ == '__main__':
    main()
