"""Freeze the approved image-level UPMC Food-101 selection without editing source."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


RULES_VERSION = 'upmc_image_clean_v1'
EXPECTED_SOURCE = {
    'UPMC-Food-101.zip':
        'bdd99f0b4dc807af6ec62c205d24b4667e7fafb5ad8bddcc0745507cf183e92c',
    'train.csv':
        '15af341c11c798cce534b2ba3eb653b432e0d5675d53330738e9c086b3463d32',
    'test.csv':
        'b3f47522bb028b2f35678f5d82aaaaa9b45d096108ddbfa14185a1668328cc4d',
}
EXPECTED_COUNTS = {
    'conflicting_groups': 405,
    'train_count': 62968,
    'test_count': 20389,
    'train_classes': 101,
    'test_classes': 101,
    'train_min_per_class': 502,
    'test_min_per_class': 158,
    'cross_split_image_groups': 0,
    'duplicate_selected_keys': 0,
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value):
    if (not value or value in ('.', '..') or '/'
            in value or '\\' in value or '\x00' in value):
        raise ValueError('UPMC CSV contains an unsafe image path component')
    return value


def build_selection_manifest(source: Path, archive: Path = None) -> dict:
    """Return one canonical decision for every original CSV row."""
    source = Path(source)
    source_sha256 = {}
    if archive is not None:
        source_sha256['UPMC-Food-101.zip'] = _sha256_file(Path(archive))
    rows = []
    original_text = []
    labels_by_image = {}
    for split in ('train', 'test'):
        csv_path = source / (split + '.csv')
        source_sha256[split + '.csv'] = _sha256_file(csv_path)
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if (reader.fieldnames is None or len(reader.fieldnames) != 4
                    or set(reader.fieldnames)
                    != {'id', 'text', 'annotation', 'label'}):
                raise ValueError('UPMC CSV columns differ from frozen source')
            for index, raw in enumerate(reader):
                name = _safe_component(raw['id'])
                annotation = _safe_component(raw['annotation'])
                label = int(raw['label'])
                if label < 0 or label >= 101:
                    raise ValueError('UPMC label is outside 0..100')
                text = raw['text']
                if text is None:
                    raise ValueError('UPMC text is missing')
                image_path = source / 'images' / split / annotation / name
                image_hash = _sha256_file(image_path)
                row_hash = hashlib.sha256(_canonical([
                    name, text, annotation, label,
                ])).hexdigest()
                rows.append({
                    'split': split,
                    'source_index': index,
                    'label': label,
                    'image_sha256': image_hash,
                    'row_sha256': row_hash,
                    'included': False,
                    'reason': '',
                })
                original_text.append(text)
                labels_by_image.setdefault(image_hash, set()).add(label)
    conflicting = {
        image_hash for image_hash, labels in labels_by_image.items()
        if len(labels) > 1
    }
    train_images = {
        row['image_sha256'] for row in rows
        if row['split'] == 'train' and row['image_sha256'] not in conflicting
    }
    seen = {'train': set(), 'test': set()}
    for row, text in zip(rows, original_text):
        split, image_hash = row['split'], row['image_sha256']
        if image_hash in conflicting:
            row['reason'] = 'label_conflict'
        elif split == 'test' and image_hash in train_images:
            row['reason'] = 'train_priority'
        else:
            key = (image_hash, text, row['label'])
            if key in seen[split]:
                row['reason'] = 'exact_duplicate'
            else:
                seen[split].add(key)
                row['included'] = True
                row['reason'] = 'included'
    selected = {
        split: [(row, original_text[index])
                for index, row in enumerate(rows)
                if row['split'] == split and row['included']]
        for split in ('train', 'test')
    }
    selected_images = {
        split: {row['image_sha256'] for row, _ in selected[split]}
        for split in ('train', 'test')
    }
    class_counts = {
        split: Counter(row['label'] for row, _ in selected[split])
        for split in ('train', 'test')
    }
    selected_keys = {
        split: [(row['image_sha256'], text, row['label'])
                for row, text in selected[split]]
        for split in ('train', 'test')
    }
    summary = {
        'conflicting_groups': len(conflicting),
        'train_count': len(selected['train']),
        'test_count': len(selected['test']),
        'train_classes': len(class_counts['train']),
        'test_classes': len(class_counts['test']),
        'train_min_per_class': min(class_counts['train'].values(), default=0),
        'test_min_per_class': min(class_counts['test'].values(), default=0),
        'cross_split_image_groups': len(
            selected_images['train'] & selected_images['test']),
        'duplicate_selected_keys': sum(
            len(selected_keys[split]) - len(set(selected_keys[split]))
            for split in ('train', 'test')
        ),
    }
    manifest = {
        'kind': 'upmc_image_clean_selection_v1',
        'rules_version': RULES_VERSION,
        'source_sha256': source_sha256,
        'summary': summary,
        'rows': rows,
    }
    manifest['manifest_sha256'] = hashlib.sha256(
        _canonical(manifest)).hexdigest()
    return manifest


def select_rows(manifest: dict, split: str) -> list:
    if split not in ('train', 'test'):
        raise ValueError('invalid split')
    if (manifest.get('kind') != 'upmc_image_clean_selection_v1'
            or manifest.get('rules_version') != RULES_VERSION):
        raise ValueError('invalid UPMC selection manifest')
    digest = manifest.get('manifest_sha256')
    payload = {key: value for key, value in manifest.items()
               if key != 'manifest_sha256'}
    if digest != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError('UPMC selection manifest hash differs')
    return [row['source_index'] for row in manifest['rows']
            if row['split'] == split and row['included']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-only', action='store_true')
    action.add_argument('--output', type=Path)
    args = parser.parse_args()
    archive = args.source.parent / 'UPMC-Food-101.zip'
    manifest = build_selection_manifest(args.source, archive)
    if manifest['source_sha256'] != EXPECTED_SOURCE:
        raise ValueError('raw UPMC source SHA-256 differs')
    if manifest['summary'] != EXPECTED_COUNTS:
        raise ValueError('UPMC image-clean census differs')
    if args.check_only:
        print(_canonical({
            **manifest['summary'],
            'manifest_sha256': manifest['manifest_sha256'],
        }).decode('utf-8'))
        return
    output = args.output
    payload = _canonical(manifest) + b'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise ValueError('selection output cannot be a symlink')
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError('selection manifest already differs')
        return
    with output.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == '__main__':
    main()
