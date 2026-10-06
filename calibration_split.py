"""Deterministic held-out calibration split for formal BiC experiments."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def build_manifest(targets, per_class=25, seed=20260722,
                   dataset='cifar100-train', excluded_indices=None,
                   sample_ids=None):
    labels = np.asarray(targets)
    excluded = set(int(index) for index in (excluded_indices or ()))
    if sample_ids is not None:
        sample_ids = [str(sample_id) for sample_id in sample_ids]
        if len(sample_ids) != len(labels):
            raise ValueError('sample_ids must have one identity per target')
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError('sample_ids must be unique')
    rng = np.random.default_rng(seed)
    by_class = {}
    for class_id in sorted(np.unique(labels).tolist()):
        candidate_indices = [
            int(index) for index in np.flatnonzero(labels == class_id)
            if int(index) not in excluded
        ]
        candidates = np.asarray(
            sorted(sample_ids[index] for index in candidate_indices)
            if sample_ids is not None else candidate_indices
        )
        if len(candidates) < per_class:
            raise ValueError(f'class {class_id} has {len(candidates)} samples')
        by_class[str(class_id)] = sorted(
            str(identity) if sample_ids is not None else int(identity)
            for identity in rng.choice(candidates, per_class, replace=False)
        )
    ordered = [identity for key in sorted(by_class, key=int) for identity in by_class[key]]
    digest = hashlib.sha256(
        json.dumps(ordered, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    manifest = {
        'dataset': str(dataset),
        'seed': int(seed),
        'per_class': int(per_class),
        'by_class': by_class,
    }
    manifest[
        'ordered_sample_ids' if sample_ids is not None else 'ordered_indices'
    ] = ordered
    manifest['sha256'] = digest
    return manifest


def manifest_indices(manifest):
    return set(int(index) for index in manifest['ordered_indices'])


def _fsync_parent(path):
    descriptor = os.open(
        path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(manifest, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
