"""Version the clean UPMC cache with four feature owners, preserving all rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


CACHE_NAME = 'upmc_food101_vfl.npz'
META_NAME = 'upmc_food101_vfl.metadata.json'
LO = np.array([0, 256, 512, 1024], dtype=np.int64)
HI = np.array([256, 512, 1024, 1536], dtype=np.int64)
NAMES = np.array(['image_a', 'image_b', 'text_a', 'text_b'])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_four_party_cache(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    source, source_meta = source_dir / CACHE_NAME, source_dir / META_NAME
    output, output_meta = output_dir / CACHE_NAME, output_dir / META_NAME
    if any(path.is_symlink() for path in (source, source_meta, output, output_meta)):
        raise ValueError('cache symlinks are forbidden')
    if not source.is_file() or not source_meta.is_file():
        raise FileNotFoundError('clean two-party source cache and metadata are required')
    metadata = json.loads(source_meta.read_text(encoding='utf-8'))
    source_hash = sha256_file(source)
    if metadata.get('output_sha256') != source_hash:
        raise ValueError('source cache hash differs from its metadata')
    with np.load(source, allow_pickle=False) as old:
        expected_keys = {'X', 'y', 'source_row_id', 'train_idx', 'test_idx',
                         'view_names', 'range_lo', 'range_hi'}
        if set(old.files) != expected_keys:
            raise ValueError('unexpected source cache fields')
        if (old['X'].ndim != 2 or old['X'].shape[1] != 1536
                or old['X'].dtype != np.float32
                or old['view_names'].tolist() != [
                    'image_resnet18_imagenet1k', 'title_tfidf_train_only']
                and old['view_names'].tolist() != ['image', 'text']
                or old['range_lo'].tolist() != [0, 512]
                or old['range_hi'].tolist() != [512, 1536]):
            raise ValueError('source cache is not the reviewed image/text layout')
        arrays = {name: old[name] for name in (
            'X', 'y', 'source_row_id', 'train_idx', 'test_idx')}
    n = arrays['X'].shape[0]
    if (any(arr.ndim != 1 for name, arr in arrays.items() if name != 'X')
            or len(arrays['y']) != n or len(arrays['source_row_id']) != n
            or len(np.unique(arrays['source_row_id'])) != n
            or len(arrays['train_idx']) + len(arrays['test_idx']) != n
            or not np.array_equal(
                np.sort(np.concatenate((arrays['train_idx'], arrays['test_idx']))),
                np.arange(n))):
        raise ValueError('source rows, IDs, or train/test indices are inconsistent')
    if not np.isfinite(arrays['X']).all():
        raise ValueError('source features are not finite')

    if output.exists() or output_meta.exists():
        if not output.is_file() or not output_meta.is_file():
            raise FileExistsError('partial four-party cache already exists')
        existing = json.loads(output_meta.read_text(encoding='utf-8'))
        if (existing.get('source_cache_sha256') != source_hash
                or existing.get('output_sha256') != sha256_file(output)
                or existing.get('views') != [
                    {'name': str(name), 'lo': int(lo), 'hi': int(hi)}
                    for name, lo, hi in zip(NAMES, LO, HI)]):
            raise FileExistsError('existing four-party cache has another identity')
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.upmc4-',
                                     suffix='.npz.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays, view_names=NAMES,
                            range_lo=LO, range_hi=HI)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        new_meta = dict(metadata)
        new_meta.update(
            dataset='UPMC Food-101 image-clean four-party v1',
            source_cache_sha256=source_hash,
            source_selection_manifest_sha256=metadata.get(
                'selection_manifest_sha256'),
            output_sha256=sha256_file(temporary),
            views=[{'name': str(name), 'lo': int(lo), 'hi': int(hi)}
                   for name, lo, hi in zip(NAMES, LO, HI)],
        )
        os.link(temporary, output)
        with output_meta.open('x', encoding='utf-8') as handle:
            json.dump(new_meta, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(build_four_party_cache(args.source_dir, args.output_dir))


if __name__ == '__main__':
    main()
